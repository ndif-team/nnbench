"""Jacobian lens (anthropics/jacobian-lens): the logit lens with a per-layer linear transport,
`unembed(norm(J_l @ h))`, read at a per-unit prompt position (the token before the would-be answer;
upstream's position rules). `transport` is None (identity), an int seed (one seeded orthogonal map
shared by every layer), "hub:<repo>@<revision>:<filename>" or "file:<path>" (a fitted lens, one map
per layer), or a tensor / {layer: tensor} mapping. Output `[n_layers, B, vocab]`."""
from __future__ import annotations

import functools
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from ..profiles import _resid
from ..tasks.lens_eval import locate
from .observations import record_resolved
from .registry import cell


def _orthogonal(d: int, seed: int) -> torch.Tensor:
    """A seeded random orthogonal [d, d] map (QR of a seeded gaussian), the same on every backend."""
    g = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


def _parse_hub(spec: str) -> tuple[str, str, str]:
    body = spec.removeprefix("hub:")
    repo_rev, sep, filename = body.partition(":")
    repo, sep2, revision = repo_rev.partition("@")
    if not (sep and sep2 and repo and revision and filename):
        raise ValueError(f"transport hub spec {spec!r} is not 'hub:<repo>@<revision>:<filename>'")
    return repo, revision, filename


@functools.lru_cache(maxsize=4)
def _load_fitted(spec: str) -> Mapping:
    """A fitted lens's per-layer {layer: [d, d]} map from a hub checkpoint or a local .pt file."""
    if spec.startswith("file:"):
        return torch.load(spec.removeprefix("file:"), map_location="cpu", weights_only=False)["J"]
    from huggingface_hub import hf_hub_download

    repo, revision, filename = _parse_hub(spec)
    path = hf_hub_download(repo, filename, revision=revision)
    return torch.load(path, map_location="cpu", weights_only=False)["J"]


def _readout_rows(h_full, positions):
    """The readout row of each unit: `[B, d]` from HF's `[B, S, d]` with one position per row, or
    `[1, d]` from vLLM's flat `[S, d]`."""
    if h_full.dim() == 3:
        index = torch.arange(h_full.shape[0], device=h_full.device)
        pos = torch.tensor(positions, device=h_full.device) % h_full.shape[1]
        return h_full[index, pos, :]
    return h_full[positions[0]].unsqueeze(0)


def _jlens_proxy(blocks, norm, head, *, layers, positions, unembed, residual, transport,
                 processor=None):
    """The J-lens readout at each unit's position, stacked over the band `[n_layers, B, vocab]`.
    Runs inside a trace."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    J = None                                    # the shared map, resolved once per trace
    with torch.no_grad():
        for i in idx:
            h = _readout_rows(_resid(blocks[i].output, residual), positions)
            if transport is not None:
                if isinstance(transport, Mapping):
                    Ji = transport[i]
                else:
                    if J is None:
                        J = (transport if isinstance(transport, torch.Tensor)
                             else _orthogonal(h.shape[-1], transport))
                        J = J.to(device=h.device, dtype=h.dtype)
                    Ji = J
                h = h @ Ji.to(device=h.device, dtype=h.dtype).T
            normed = norm(h)
            if unembed == "logits_processor":
                rows.append(processor(head, normed))
            else:
                rows.append(F.linear(normed, head.weight) if unembed == "weight" else head(normed))
    return torch.stack(rows, dim=0)


def _positions(model, units, position):
    """Each unit's readout position in its own token sequence; a negative index counts from the
    end, which a left-padded batch leaves valid."""
    return [locate(model.tokenizer, unit, position) for unit in units]


def _padded_positions(model, units, positions):
    """HF: the positions in a left-padded batch of `units`, whose rows end at the same column; a
    negative position counts from that end and needs no offset."""
    if all(pos < 0 for pos in positions):
        return positions
    counts = [len(model.tokenizer.encode(unit)) for unit in units]
    width = max(counts)
    return [pos if pos < 0 else width - count + pos for pos, count in zip(positions, counts)]


@cell("jacobian_lens", family="*", backend="hf", realizations=[{"unembed": "weight"}])
def jacobian_lens_hf(be, model, m, units, *, layers="all", position="last", unembed="module",
                     residual="plain", transport=None):
    if isinstance(transport, str):
        transport = _load_fitted(transport)
    positions = _padded_positions(model, units, _positions(model, units, position))

    def build(rows):
        return _jlens_proxy(m.blocks(model), m.norm(model), m.head(model), layers=layers,
                            positions=[positions[r] for r in rows], unembed=unembed,
                            residual=residual, transport=transport)

    return be.run(model, units, build)


@cell("jacobian_lens", family="*", backend="vllm_async",
      realizations=[{"unembed": "module"}, {"unembed": "weight"}])
def jacobian_lens_vllm(be, model, m, units, *, layers="all", position="last",
                       unembed="logits_processor", residual=None, transport=None):
    if isinstance(transport, str):
        transport = _load_fitted(transport)
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    positions = _positions(model, units, position)

    def build(rows):
        return _jlens_proxy(m.blocks(model), m.norm(model), m.head(model), layers=layers,
                            positions=[positions[r] for r in rows], unembed=unembed,
                            residual=residual, transport=transport,
                            processor=m.logits_processor(model))

    return be.run(model, units, build)

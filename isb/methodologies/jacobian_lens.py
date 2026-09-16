"""Jacobian lens (anthropics/jacobian-lens, ICLR-track 2026) — family-generic cell (§12.8).

The J-lens readout is the logit lens with one extra matmul: transport the residual at layer l
through a per-layer linear map J_l BEFORE the final norm + unembed,

    lens_l(h) = unembed( norm( J_l @ h ) )        (transport=None -> identity -> plain logit lens)

read at a PER-ITEM prompt position (the token preceding the would-be answer; upstream README).
Upstream, J_l is the fitted average input-output Jacobian E[dh_final/dh_l]; fitting needs autograd
(offline, HF-side — on vLLM `grad` is ERROR, the known frontier). The BENCH question is not the
map's faithfulness but whether the serving backend reproduces the readout: read-at-position +
transport matmul (meta-compute) + norm + portable unembed, HF vs vLLM, on the real J-lens datasets
(`data/jlens/`, spec `jacobian_lens_gpt2`).

`transport` selects the map, in order of realism:
  - None: identity, the plain logit lens at the readout position.
  - int seed: a SEEDED RANDOM ORTHOGONAL matrix (one shared [d, d] map), built deterministically
    once per trace on first use (same seed -> the same matrix on every backend, so the oracle
    compares like with like; the QR setup cost is in the timed region, unlike a fitted tensor, so
    transport-task perf reads as matmul + one QR). Full-rank and norm-preserving: exercises the
    exact J-matmul code path without the fitted artifact.
  - "hub:<repo>@<revision>:<filename>": the REAL fitted lens (the `trained` tag): downloads the
    upstream checkpoint (cached), whose `J` is one [d, d] matrix PER LAYER; resolved once per
    process (lru_cache), passed into the trace as a mapping.
  - "file:<path>": a locally fitted lens in the same .pt layout — e.g. the maps a
    jacobian_collect run produced, exported by scripts/export_jacobian.py.
  - a torch.Tensor (one shared map) or a Mapping {layer_index: tensor} (per-layer), directly.

Variances (params): `layers` ("all" | list) — the read band; `position` ("last" | "last_newline",
the upstream per-dataset readout rules, resolved per item via the tokenizer); `unembed`
("module" idiomatic | "weight" portable — vLLM guards lm_head.forward); `residual` (None -> the
family's per-engine denotation; "plain"/"fused" override); `transport` (above).
"""
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
    """Deterministic random orthogonal [d, d] map (QR of seeded gaussian, CPU fp32): the seeded
    stand-in for a fitted J_l. Same seed -> identical matrix on every backend and process."""
    g = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q


def _parse_hub(spec: str) -> tuple[str, str, str]:
    """"hub:<repo>@<revision>:<filename>" -> (repo, revision, filename). Loud on malformed input."""
    body = spec.removeprefix("hub:")
    repo_rev, sep, filename = body.partition(":")
    repo, sep2, revision = repo_rev.partition("@")
    if not (sep and sep2 and repo and revision and filename):
        raise ValueError(f"transport hub spec {spec!r} is not 'hub:<repo>@<revision>:<filename>'")
    return repo, revision, filename


@functools.lru_cache(maxsize=4)
def _load_fitted(spec: str) -> Mapping:
    """Load a fitted lens; returns its per-layer {layer: [d, d]} map. Two schemes, one .pt
    layout ({"J": {layer_index: tensor}, ...}): "hub:<repo>@<revision>:<filename>" downloads
    (cached) an upstream checkpoint; "file:<path>" reads a local artifact, e.g. one exported
    from a jacobian_collect run by scripts/export_jacobian.py."""
    if spec.startswith("file:"):
        return torch.load(spec.removeprefix("file:"), map_location="cpu",
                          weights_only=False)["J"]
    from huggingface_hub import hf_hub_download

    repo, revision, filename = _parse_hub(spec)
    path = hf_hub_download(repo, filename, revision=revision)
    return torch.load(path, map_location="cpu", weights_only=False)["J"]


def _jlens_proxy(blocks, norm, head, *, layers, pos, unembed, residual, transport):
    """Stack the J-lens readout at `pos` over the band. Runs INSIDE a trace. The per-backend caller
    passes `take(h_full)` semantics via pos: hf tensors are [B, S, D], vllm flat [S, D]; both keep
    the seq dim at -2, so integer `pos` indexes the token uniformly and unsqueeze restores a row
    dim on the flat layout."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    J = None                                    # shared map resolved ONCE per trace, on first use
    with torch.no_grad():                       # forward-only; required on vLLM, harmless on HF
        for i in idx:
            h_full = _resid(blocks[i].output, residual)
            h = h_full[..., pos, :]             # [B, D] on hf, [D] on vllm
            if h.dim() == 1:
                h = h.unsqueeze(0)              # flat layout -> [1, D], matching be.last's row shape
            if transport is not None:
                if isinstance(transport, Mapping):      # fitted lens: one map PER LAYER; a band
                    Ji = transport[i]                   # layer outside the lens raises KeyError, loud
                else:
                    if J is None:               # one QR + one device/dtype cast per trace, not per layer
                        J = (transport if isinstance(transport, torch.Tensor)
                             else _orthogonal(h.shape[-1], transport))
                        J = J.to(device=h.device, dtype=h.dtype)
                    Ji = J
                h = h @ Ji.to(device=h.device, dtype=h.dtype).T
            normed = norm(h)
            rows.append(F.linear(normed, head.weight) if unembed == "weight" else head(normed))
    return torch.stack(rows, dim=0)             # [n_band_layers, 1-or-B, vocab]


@cell("jacobian_lens", family="*", backend="hf")
def jacobian_lens_hf(be, model, m, prompts, *, layers="all", position="last",
                     unembed="module", residual="plain", transport=None):
    if isinstance(transport, str):                        # fitted-lens spec -> per-layer map (cached)
        transport = _load_fitted(transport)
    pos = locate(model.tokenizer, prompts[0], position)   # per-item, input-space (upstream rules)

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _jlens_proxy(m.blocks(model), m.norm(model), m.head(model),
                            layers=layers, pos=pos, unembed=unembed,
                            residual=residual, transport=transport)
    return be.run(model, prompts, build)


@cell("jacobian_lens", family="*", backend="vllm_async")
def jacobian_lens_vllm(be, model, m, prompts, *, layers="all", position="last",
                       unembed="weight", residual=None, transport=None):
    if isinstance(transport, str):                        # fitted-lens spec -> per-layer map (cached)
        transport = _load_fitted(transport)
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    pos = locate(model.tokenizer, prompts[0], position)

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _jlens_proxy(m.blocks(model), m.norm(model), m.head(model),
                            layers=layers, pos=pos, unembed=unembed,
                            residual=residual, transport=transport)
    return be.run(model, prompts, build)

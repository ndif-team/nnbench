"""Activation patching (causal tracing): capture a layer's residual stream from the clean prompts,
replace the corrupted prompts' stream at that layer with it, read the corrupted run's next-token
logits. Two traces (`be.patch`): the documented vLLM form, where invokes share no values. A unit is
a length-matched (clean, corrupted) pair, so the streams align position by position; `positions`
patches every position or the last one. `residual` selects the stream on vLLM's fused-residual
families, where the layer returns (hidden, residual) and the clean stream is written back as
(clean - residual, residual)."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import final_logits
from .registry import cell


def _capture(blocks, layer, residual):
    """The clean residual stream at `layer`, an independent snapshot. Runs inside trace 1."""
    with torch.no_grad():
        return _resid(blocks[layer].output, residual).clone()


def _read_final(model, head, site):
    """The corrupted run's own next-token logits (the unpatched baseline)."""
    with torch.no_grad():
        return final_logits(model, head, site)


def _patch_and_read(blocks, model, head, *, layer, clean_act, residual, site, positions="all"):
    """Replace `layer`'s stream with the clean snapshot and read the next-token logits. Runs inside
    trace 2."""
    with torch.no_grad():
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        clean = clean_act.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(clean.shape) != tuple(hidden.shape):
            raise ValueError(
                f"patch shape {tuple(clean.shape)} != target {tuple(hidden.shape)}; clean/corrupted "
                f"token lengths must match for position-aligned patching")
        if positions == "last":
            stream = _resid(out, residual)
            clean = torch.cat([stream[..., :-1, :], clean[..., -1:, :]], dim=-2)
        elif positions != "all":
            raise ValueError(f"unknown patch positions {positions!r} (expected 'all' or 'last')")
        if residual == "fused":
            if not (is_tuple and len(out) >= 2):
                raise ValueError("fused patch needs a (hidden, residual) layer output")
            blocks[layer].output = (clean - out[1], *out[1:])
        else:
            blocks[layer].output = (clean, *out[1:]) if is_tuple else clean
        return final_logits(model, head, site)


def _patch_cell(be, model, m, units, *, layer, residual, patch, site, positions="all"):
    clean = [unit[0] for unit in units]
    corrupted = [unit[1] for unit in units]
    h, head = m.blocks(model), m.head(model)
    if not patch:                                        # baseline: the corrupted runs, no transplant
        def build(rows):
            return _read_final(model, head, site)

        return be.run(model, corrupted, build)

    def capture():
        return _capture(h, layer, residual)

    def patch_fn(clean_act):
        return _patch_and_read(h, model, head, layer=layer, clean_act=clean_act, residual=residual,
                               site=site, positions=positions)

    return be.patch(model, clean, corrupted, capture=capture, patch=patch_fn)


@cell("activation_patching", family="*", backend="hf")
def patch_hf(be, model, m, units, *, layer=6, residual="plain", patch=True, positions="all"):
    return _patch_cell(be, model, m, units, layer=layer, residual=residual, patch=patch,
                       site="head", positions=positions)


@cell("activation_patching", family="*", backend="vllm_async", realizations=[{"residual": "plain"}])
def patch_vllm(be, model, m, units, *, layer=6, residual=None, patch=True, positions="all"):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    return _patch_cell(be, model, m, units, layer=layer, residual=residual, patch=patch,
                       site="engine", positions=positions)

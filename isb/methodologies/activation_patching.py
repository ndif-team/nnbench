"""Activation patching (causal tracing) — fixed per-cell methodology (design.md §12; L1 motif
"activation patching / causal tracing").

Cross-prompt write methodology: capture a layer's residual from a CLEAN run, inject it into a
CORRUPTED run at the same layer, observe the corrupted run's next-token distribution. We use the
**two-trace** formulation (capture in trace 1, inject in trace 2 via `be.patch`). This is the
DOCUMENTED vLLM recipe, not an invention: the canonical single-trace form uses `tracer.barrier(2)`
to hand the clean value from invoke 1 to invoke 2, but on vLLM the barrier is not shared across
invokes (cross-invoke dependencies are unsupported — see nnsight VLLM_GUIDE "Activation Patching"
and docs/developing/barrier-vllm-not-shared.md). Multi-invoke itself IS supported for *independent*
prompts; it's the cross-invoke value hand-off that isn't. Two separate single-prompt traces
guarantee ordering and need no barrier, so the methodology is the same shape on both backends.

`clean` and `corrupted` are a length-matched minimal pair (differ at one token), so (a) the residual
shapes align for replacement and (b) the patch actually changes the corrupted output — without that
effect the SUPPORTED verdict would be vacuous (cf. the steering effect-size guard, F-6).

The patch uses whole-tuple **replacement** (the vLLM-safe write form; in-place raises on vLLM
inference tensors, F-5). Variances (params): `layer` (which block's residual to transplant),
`residual` ("plain" | "fused" — same fused-residual reconstruction as logit-lens F-7, so this ports
to vLLM-Llama too).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .logit_lens import _resid
from .registry import cell


def _capture(blocks, layer, residual):
    """Trace-1 build: snapshot the clean residual stream at `layer`. Returns a proxy (be saves it)."""
    with torch.no_grad():
        return _resid(blocks[layer].output, residual).clone()  # clone -> an independent snapshot


def _patch_and_read(blocks, norm, head, *, layer, clean_act, residual, last_fn):
    """Trace-2 build: replace `layer`'s residual with the clean snapshot, read final logits."""
    with torch.no_grad():
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        clean = clean_act.to(device=hidden.device, dtype=hidden.dtype)  # runtime device/dtype
        if tuple(clean.shape) != tuple(hidden.shape):
            raise ValueError(
                f"patch shape {tuple(clean.shape)} != target {tuple(hidden.shape)} — clean/corrupted "
                f"token lengths must match for position-aligned patching"
            )
        blocks[layer].output = (clean, *out[1:]) if is_tuple else clean  # whole-tuple replace (vLLM-safe)

        normed = norm(_resid(blocks[-1].output, residual))      # final residual -> final norm
        logits = F.linear(normed, head.weight)                  # portable unembed (lm_head guarded)
    return last_fn(logits)


@cell("activation_patching", family="gpt2", backend="hf")
def patch_gpt2_hf(be, model, prompts, *, layer=6, residual="plain"):
    clean, corrupted = prompts
    h, ln_f, head = model.transformer.h, model.transformer.ln_f, model.lm_head
    return be.patch(
        model, clean, corrupted,
        capture=lambda: _capture(h, layer, residual),
        patch=lambda clean_act: _patch_and_read(
            h, ln_f, head, layer=layer, clean_act=clean_act, residual=residual, last_fn=be.last),
    )


@cell("activation_patching", family="gpt2", backend="vllm_async")
def patch_gpt2_vllm(be, model, prompts, *, layer=6, residual="plain"):
    clean, corrupted = prompts
    h, ln_f, head = model.transformer.h, model.transformer.ln_f, model.lm_head
    return be.patch(
        model, clean, corrupted,
        capture=lambda: _capture(h, layer, residual),
        patch=lambda clean_act: _patch_and_read(
            h, ln_f, head, layer=layer, clean_act=clean_act, residual=residual, last_fn=be.last),
    )

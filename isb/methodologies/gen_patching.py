"""Generation-time cross-prompt patching — the transplant edge run UNDER the decode loop.

In the program-model vocabulary (design.md §3.5) this is the **step-lift of the transplant edge**:
activation patching (transplant: read in run A → write in run B) lifted into a multi-token greedy
generation (the step quantifier). It is a COMPOSITION cell — it adds no new primitive; the boundary
read, the replacement write, the two-trace transplant, and bounded iteration are each already
measured. Its two jobs:

  1. **Law check (design.md §3.7).** Does the transplant edge stay valid when run during a decode
     loop? `∧(footprint)` predicts SUPPORTED-degraded (every entry is SUPPORTED, bf16 a near-tie
     like the single-forward patch); the cell tests whether the composition actually holds. This is
     the transplant-edge analogue of what generation-time steering confirmed for the injection edge.
  2. **The causalab `locate` recipe.** locate scores a cross-prompt interchange on *generated*
     tokens; this cell is that footprint, the recipe the Macro-tier port needs.

It is NOT a novel KV-cache frontier. The patch is injected at the base prompt's PREFILL; decode
steps don't recompute prompt positions, so the patched residual is simply part of the forward that
builds this request's own (intra-request) KV cache — ordinary dataflow. nnsight disables the
cross-request prefix cache, so there is no cache-skip path that could silently drop it.

Mechanics: capture the whole residual at `layer` from the CLEAN (source) prompt; greedily decode
the CORRUPTED (base) prompt, injecting that residual at the prefill step (whole-tuple replacement,
the vLLM-safe write — in-place raises on inference tensors); read each step's next-token logits.
`clean`/`corrupted` are a length-matched minimal pair so the residuals align for replacement and the
transplant actually moves the generated continuation (else the verdict is vacuous — the effect-size
guard catches it).

Oracle caveat (honest): per-step logits are scored against the HF reference with the dtype control
separating bf16 precision from a real mechanism failure. Cross-engine greedy *trajectories* can fork
on a near-tie token for engine-numeric reasons alone, so the fp32 re-run is the mechanism verdict
and the pair is chosen for a decisive transplant; bf16 is reported as its own degradation.

Variances (params): `layer` (which block's residual to transplant), `residual` ("plain" | "fused"
— the same fused-residual reconstruction as the logit-lens/patching cells, for a vLLM-Llama port),
`new_tokens`, `bound` ("bounded" `iter[0:N]` — the vLLM working idiom — vs "unbounded" `iter[:]`,
the unbounded-iteration saves-drop frontier marker). `patch=False` is the unpatched-generation
baseline (same decode loop, no transplant), so the same cell yields the effect-size control.
"""
from __future__ import annotations

import torch

from .logit_lens import _resid
from .registry import cell


def _capture(blocks, layer, residual):
    """Trace-1 build: snapshot the clean residual stream at `layer` (an independent clone)."""
    with torch.no_grad():
        return _resid(blocks[layer].output, residual).clone()


def _inject_at_prefill(blocks, layer, clean_act):
    """Replace `layer`'s residual with the captured clean snapshot — but ONLY at the prefill step,
    detected by shape: the prefill residual spans the whole prompt and matches the captured shape,
    while each decode step's residual is length-1 (the new token alone) and is left untouched. The
    transplant therefore lands once, at prefill, and rides the request's own KV cache from there.
    Whole-tuple replacement (the vLLM-safe write; in-place raises on inference tensors)."""
    with torch.no_grad():
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        clean = clean_act.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(clean.shape) == tuple(hidden.shape):          # prefill: full-prompt residual
            blocks[layer].output = (clean, *out[1:]) if is_tuple else clean


def _check_bound(bound):
    if bound not in ("bounded", "unbounded"):
        raise ValueError(f"unknown iteration bound {bound!r} (expected 'bounded' or 'unbounded')")


@cell("gen_patching", family="gpt2", backend="hf")
def gen_patching_gpt2_hf(be, model, prompts, *, layer=9, residual="plain",
                         bound="bounded", new_tokens=8, patch=True):
    _check_bound(bound)
    clean, corrupted = prompts
    h = model.transformer.h

    if not patch:                                              # unpatched-generation baseline
        return be.generate(model, [corrupted], lambda: model.lm_head.output[:, -1, :],
                           new_tokens=new_tokens, bounded=(bound == "bounded"))

    def step(clean_act):
        _inject_at_prefill(h, layer, clean_act)
        return model.lm_head.output[:, -1, :]                 # this step's next-token logits

    return be.generate_patch(model, clean, corrupted,
                             capture=lambda: _capture(h, layer, residual),
                             build_step=step, new_tokens=new_tokens, bounded=(bound == "bounded"))


@cell("gen_patching", family="gpt2", backend="vllm_async")
def gen_patching_gpt2_vllm(be, model, prompts, *, layer=9, residual="plain",
                           bound="bounded", new_tokens=8, patch=True):
    _check_bound(bound)
    clean, corrupted = prompts
    h = model.transformer.h

    if not patch:
        return be.generate(model, [corrupted], lambda: model.logits[-1:, :],
                           new_tokens=new_tokens, bounded=(bound == "bounded"))

    def step(clean_act):
        _inject_at_prefill(h, layer, clean_act)
        return model.logits[-1:, :]                            # engine site == portable unembed

    return be.generate_patch(model, clean, corrupted,
                             capture=lambda: _capture(h, layer, residual),
                             build_step=step, new_tokens=new_tokens, bounded=(bound == "bounded"))

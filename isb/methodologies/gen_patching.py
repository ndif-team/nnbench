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


def _inject_transplant(blocks, layer, clean_act):
    """Replace `layer`'s residual with the captured clean snapshot (whole-tuple replacement — the
    vLLM-safe write; in-place raises on inference tensors). The cell calls this exactly once, on the
    PREFILL forward (the first generation forward, which holds the whole prompt); the transplant then
    rides the request's own KV cache through the decode steps. The prefill step is identified by the
    caller's first-forward flag, NOT by shape — a shape test would misfire on a one-token prompt,
    whose decode steps share the prefill's [1, hidden] shape and would be re-patched every step.

    Raises if the snapshot's shape does not match the residual being replaced — i.e. the
    clean/corrupted prompts are not length-matched (or the prefill was chunked), which makes a
    position-aligned whole-residual replacement ill-defined; failing loudly beats a silent no-op."""
    with torch.no_grad():
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        clean = clean_act.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(clean.shape) != tuple(hidden.shape):
            raise ValueError(
                f"transplant shape {tuple(clean.shape)} != prefill residual {tuple(hidden.shape)} — "
                f"clean/corrupted prompts must be length-matched for position-aligned patching"
            )
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
        def step():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
            return model.lm_head.output[:, -1, :]
        return be.generate(model, [corrupted], step,
                           new_tokens=new_tokens, bounded=(bound == "bounded"))

    injected = [False]                                         # inject once, on the prefill forward

    def capture():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _capture(h, layer, residual)

    def step(clean_act):
        if not injected[0]:                                    # the first generation forward IS prefill
            _inject_transplant(h, layer, clean_act)
            injected[0] = True
        return model.lm_head.output[:, -1, :]                 # this step's next-token logits

    return be.generate_patch(model, clean, corrupted, capture=capture,
                             build_step=step, new_tokens=new_tokens, bounded=(bound == "bounded"))


@cell("gen_patching", family="gpt2", backend="vllm_async")
def gen_patching_gpt2_vllm(be, model, prompts, *, layer=9, residual="plain",
                           bound="bounded", new_tokens=8, patch=True):
    _check_bound(bound)
    clean, corrupted = prompts
    h = model.transformer.h

    if not patch:
        def step():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
            return model.logits[-1:, :]
        return be.generate(model, [corrupted], step,
                           new_tokens=new_tokens, bounded=(bound == "bounded"))

    injected = [False]                                         # inject once, on the prefill forward

    def capture():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _capture(h, layer, residual)

    def step(clean_act):
        if not injected[0]:                                    # the first generation forward IS prefill
            _inject_transplant(h, layer, clean_act)
            injected[0] = True
        return model.logits[-1:, :]                            # engine site == portable unembed

    return be.generate_patch(model, clean, corrupted, capture=capture,
                             build_step=step, new_tokens=new_tokens, bounded=(bound == "bounded"))

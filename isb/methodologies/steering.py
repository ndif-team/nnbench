"""Steering (activation addition / ActAdd) — fixed per-cell methodology (design.md §12;
registry entry "steering / ActAdd").

This is the first *write* methodology, so it tests the backend's **write fidelity** — the open
question for vLLM. We add a steering vector into a block's residual-stream output and then read the
model's resulting next-token distribution. The equivalence oracle resolves the three outcomes that
a crash-or-not check cannot tell apart:

  - vLLM applies the write -> steered logits match HF-steered            -> SUPPORTED
  - vLLM silently drops the write (no-op) -> logits == unsteered baseline
    -> diverge from HF-steered                                           -> SILENTLY_WRONG  (the dangerous cell)
  - vLLM raises on the write (e.g. in-place update of an inference tensor) -> ERROR

Observable = the **portable** unembed (weight matmul; `lm_head.forward` is guarded on vLLM, so
unembed must use the weight matmul) of the FINAL block's residual, last token. We steer an *earlier* block (`layer` <
last) so the perturbation propagates through the remaining layers into that readout, and the write
and the read land on different modules — no write-then-read-same-module ordering hazard.

Variances (params, §12):
  - `layer`  : which block to steer (abstract index into the model's own block list).
  - `target` : token whose unembed row is the steering DIRECTION (interpretable: "steer toward T").
  - `alpha`  : RELATIVE strength = multiples of the residual's own per-token norm. Self-calibrating,
               so there is no hard-coded magnitude; `alpha=0` is the unsteered baseline.
  - `mode`   : "inplace" (`hidden[:] = ...`, the vLLM-fragile form) vs "replace" (whole-tuple,
               builds a NEW tensor) — the divergence the gotcha cheat-sheet + tribal knowledge flag.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import cell


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _resolve_token(tokenizer, target: str) -> int:
    """Token id whose unembed row gives the steering direction. Same tokenizer on both backends."""
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"steering target {target!r} tokenized to nothing")
    return ids[-1]  # most content-bearing piece for a multi-token target


def _steer_and_read(blocks, norm, head, *, layer, token_id, alpha, mode, last_fn):
    """Steer blocks[layer], read the final block's portable-unembed logits. Runs INSIDE a trace.

    `alpha=0` performs no write at all (the honest unsteered baseline), so the same cell yields the
    control the effect-size check needs — run in the SAME `mode` as the cell under test.

    Reuse contract (the invariants the readout assumes — assert/keep these if reusing for new
    families): the residual stream is tuple element `[0]` (true for GPT-2/HF blocks); the final
    block's output is the PRE-final-norm residual (so `norm(...)` then a weight matmul reproduces
    the model's logits); and `layer` is strictly before the read-out (final) block, enforced below
    so the write and the read never collide on the same module.
    """
    n = len(blocks)
    if alpha != 0 and layer % n == n - 1:
        raise ValueError(
            f"steer layer {layer} is the read-out (final) block; pick layer < {n - 1} so the write "
            f"and the portable-unembed read land on different modules (no same-module write-then-read)"
        )
    with torch.no_grad():                                   # forward-only aux compute (vLLM activations
        # are inference tensors); note no_grad does NOT permit the in-place write below — that hits
        # InferenceMode protection on vLLM and is the ERROR verdict for mode="inplace".
        direction = F.normalize(head.weight[token_id].float(), dim=0).to(head.weight.dtype)

        if alpha != 0:
            out = blocks[layer].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            scale = hidden.norm(dim=-1).mean()              # residual's own scale -> alpha is relative
            vec = (alpha * scale) * direction
            if mode == "inplace":
                hidden[:] = hidden + vec                    # in-place into the live buffer (vLLM-fragile)
            elif mode == "replace":
                new_hidden = hidden + vec
                blocks[layer].output = (new_hidden, *out[1:]) if is_tuple else new_hidden
            else:
                raise ValueError(f"unknown steering mode {mode!r}")

        normed = norm(_untuple(blocks[-1].output))          # final residual -> final norm
        logits = F.linear(normed, head.weight)              # portable unembed (lm_head.forward guarded)
    return last_fn(logits)


@cell("steering", family="gpt2", backend="hf")
def steering_gpt2_hf(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, mode="inplace"):
    token_id = _resolve_token(model.tokenizer, target)
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _steer_and_read(
            model.transformer.h, model.transformer.ln_f, model.lm_head,
            layer=layer, token_id=token_id, alpha=alpha, mode=mode, last_fn=be.last,
        )
    return be.run(model, prompts, build)


@cell("steering", family="gpt2", backend="vllm_async")
def steering_gpt2_vllm(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, mode="inplace"):
    token_id = _resolve_token(model.tokenizer, target)
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _steer_and_read(
            model.transformer.h, model.transformer.ln_f, model.lm_head,
            layer=layer, token_id=token_id, alpha=alpha, mode=mode, last_fn=be.last,
        )
    return be.run(model, prompts, build)

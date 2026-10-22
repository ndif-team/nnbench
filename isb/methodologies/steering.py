"""Steering (activation addition / ActAdd) — fixed per-cell methodology (design.md §12;
registry entry "steering / ActAdd").

This is the first *write* methodology, so it tests the backend's **write fidelity** — the open
question for vLLM. We add a steering vector into a block's residual-stream output and then read the
model's resulting next-token distribution. The equivalence oracle resolves the three outcomes that
a crash-or-not check cannot tell apart:

  - vLLM applies the write -> steered logits match HF-steered            -> SUPPORTED
  - vLLM silently drops the write (no-op) -> logits == unsteered baseline
    -> diverge from HF-steered                                           -> SILENTLY_WRONG  (the dangerous cell)
  - vLLM raises on the write -> ERROR (in-place updates of inference tensors raised on nnsight
    versions before 0.8; on 260c555 both write forms score SUPPORTED)

Observable = the model's own next-token logits at the documented site (`readout.final_logits`:
HF `lm_head.output[:, -1, :]`, vLLM `model.logits`), last token. We steer an *earlier* block
(`layer` < last) so the perturbation propagates through the remaining layers into that readout.
The direction is built before the trace (`readout.head_direction` on HF,
`readout.checkpoint_direction` on vLLM, whose `lm_head.weight` is a rank slice under tensor
parallelism) and moved onto the stream inside the block.

Variances (params, §12):
  - `layer`  : which block to steer (abstract index into the model's own block list).
  - `target` : token whose unembed row is the steering DIRECTION (interpretable: "steer toward T").
  - `alpha`  : RELATIVE strength = multiples of the residual's own per-token norm. Self-calibrating,
               so there is no hard-coded magnitude; `alpha=0` is the unsteered baseline.
  - `mode`   : "inplace" (`hidden[:] = ...`, the live buffer) vs "replace" (whole-tuple,
               builds a NEW tensor) — the divergence the gotcha cheat-sheet + tribal knowledge flag.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid        # the documented read pattern: plain vs fused (out[0]+out[1])
from .observations import record_resolved
from .readout import checkpoint_direction, checkpoint_of, final_logits, head_direction
from .registry import cell


def _resolve_token(tokenizer, target: str) -> int:
    """Token id whose unembed row gives the steering direction. Same tokenizer on both backends."""
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"steering target {target!r} tokenized to nothing")
    return ids[-1]  # most content-bearing piece for a multi-token target


def _strength_scale(stream, scale_by):
    """What `alpha` multiplies: the stream's per-token norm averaged over the forward's tokens
    ("mean_norm"), or each token's own norm ("token_norm", the norm-matched form)."""
    if scale_by == "mean_norm":
        return stream.norm(dim=-1).mean()
    if scale_by == "token_norm":
        return stream.norm(dim=-1, keepdim=True)
    raise ValueError(f"unknown steering scale {scale_by!r} (expected 'mean_norm' or 'token_norm')")


def _steer_and_read(blocks, model, head, *, layer, direction, alpha, mode, site, residual="plain",
                    scale_by="mean_norm"):
    """Steer blocks[layer] along `direction`, read the next-token logits at `site`
    (`readout.final_logits`). Runs INSIDE a trace.

    `direction` is a unit vector built before the trace; it is moved onto the steered stream's
    device and dtype here (the documented form for a tensor referenced from outside the block).
    `alpha=0` performs no write at all (the honest unsteered baseline), so the same cell yields the
    control the effect-size check needs — run in the SAME `mode` as the cell under test.

    `residual` selects the stream the strength is relative to: "plain" is `out[0]`; "fused" is
    `out[0] + out[1]`, the true stream on vLLM fused-residual RMSNorm families (Llama/Qwen/
    NemotronH/...), where a block's output is `(hidden, residual)`. Falls back to plain on a bare
    tensor, so HF/GPT-2 are unaffected.

    `layer` must be strictly before the final block (enforced below): steering the last block
    leaves no layer for the write to propagate through before the readout.
    """
    n = len(blocks)
    if alpha != 0 and layer % n == n - 1:
        raise ValueError(
            f"steer layer {layer} is the read-out (final) block; pick layer < {n - 1} so the write "
            f"propagates through at least one block before the readout"
        )
    with torch.no_grad():                                   # forward-only aux compute (vLLM activations
        # are inference tensors). The in-place write below raised under InferenceMode on nnsight
        # versions before 0.8; on 260c555 it scores SUPPORTED.
        if alpha != 0:
            out = blocks[layer].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            # the stream's own scale -> alpha is relative. On fused-residual vLLM layers the stream
            # is out[0] + out[1]; out[0] alone is the layer's sub-block output.
            scale = _strength_scale(_resid(out, residual), scale_by)
            vec = (alpha * scale) * direction.to(device=hidden.device, dtype=hidden.dtype)
            if mode == "inplace":
                hidden[:] = hidden + vec                    # in-place into the live buffer
            elif mode == "replace":
                new_hidden = hidden + vec
                blocks[layer].output = (new_hidden, *out[1:]) if is_tuple else new_hidden
            else:
                raise ValueError(f"unknown steering mode {mode!r}")
        return final_logits(model, head, site)


@cell("steering", family="*", backend="hf", realizations=[{"mode": "inplace"}])
def steering_hf(be, model, m, prompts, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
                scale_by="mean_norm"):
    direction = (head_direction(model, m.head(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)               # alpha=0 writes nothing
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _steer_and_read(
            m.blocks(model), model, m.head(model),
            layer=layer, direction=direction, alpha=alpha, mode=mode, site="head",
            scale_by=scale_by,
        )
    return be.run(model, prompts, build)


@cell("steering", family="*", backend="vllm_async", realizations=[{"mode": "inplace"}])
def steering_vllm(be, model, m, prompts, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
                  residual=None, scale_by="mean_norm"):
    # The read-out residual default derives from the family's vLLM denotation (fused-residual
    # RMSNorm families must read hidden+residual; plain drops the accumulated residual). Matches
    # the previous per-family defaults exactly: gpt2 plain, llama/nemotron fused. An explicit
    # residual= param still overrides.
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    direction = (checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)               # alpha=0 writes nothing
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _steer_and_read(
            m.blocks(model), model, m.head(model),
            layer=layer, direction=direction, alpha=alpha, mode=mode, site="engine",
            residual=residual, scale_by=scale_by,
        )
    return be.run(model, prompts, build)

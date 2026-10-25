"""Steering (activation addition): add a steering vector into a block's residual-stream output and
read the model's next-token logits. The direction is the target token's unit unembedding row, built
before the trace; `alpha` is relative to the stream's own per-token norm; `alpha=0` writes nothing.
`mode` selects the write form, in place into the live buffer or whole-tuple replacement."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import checkpoint_direction, checkpoint_of, final_logits, head_direction
from .registry import cell


def _resolve_token(tokenizer, target: str) -> int:
    """Token id whose unembed row gives the steering direction."""
    ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"steering target {target!r} tokenized to nothing")
    return ids[-1]  # most content-bearing piece for a multi-token target


def _attention_mask(model):
    """A padded HF batch's attention mask `[B, S]`, read from the root model's inputs, or None for
    a batch without padding: transformers 5.18 and later drop an all-ones mask inside `generate()`.
    Read it before any block's output: nnsight runs the trace body alongside the forward and
    refuses a module's input once the model has run past it."""
    args, kwargs = model.inputs
    return kwargs.get("attention_mask")


def _strength_scale(stream, scale_by, mask=None):
    """What `alpha` multiplies, per unit: the stream's per-token norm averaged over the unit's real
    tokens ("mean_norm"; `mask` marks them in a left-padded batch, its rightmost columns being the
    stream's positions) or each token's own norm ("token_norm"). Shaped to broadcast over the
    stream: `[B, 1, 1]` for `[B, S, d]`, `[1, 1]` for vLLM's flat `[S, d]`."""
    norms = stream.norm(dim=-1)
    if scale_by == "token_norm":
        return norms.unsqueeze(-1)
    if scale_by != "mean_norm":
        raise ValueError(f"unknown steering scale {scale_by!r} (expected 'mean_norm' or 'token_norm')")
    if mask is None:
        return norms.mean(dim=-1, keepdim=True).unsqueeze(-1)
    m = mask[..., mask.shape[-1] - norms.shape[-1]:].to(norms.device, norms.dtype)
    return ((norms * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)).unsqueeze(-1)


def _steer_and_read(blocks, model, head, *, layer, direction, alpha, mode, site, residual="plain",
                    scale_by="mean_norm", masked=False):
    """Steer blocks[layer] along `direction` and read the next-token logits at `site`. Runs inside
    a trace. `masked` reads the padded batch's attention mask for the per-unit mean norm (HF)."""
    n = len(blocks)
    if alpha != 0 and layer % n == n - 1:
        raise ValueError(
            f"steer layer {layer} is the read-out (final) block; pick layer < {n - 1} so the write "
            f"propagates through at least one block before the readout")
    with torch.no_grad():
        if alpha != 0:
            mask = _attention_mask(model) if masked else None     # before the model runs past it
            out = blocks[layer].output
            is_tuple = isinstance(out, tuple)
            hidden = out[0] if is_tuple else out
            stream = _resid(out, residual)
            scale = _strength_scale(stream, scale_by, mask)
            vec = (alpha * scale) * direction.to(device=hidden.device, dtype=hidden.dtype)
            if mode == "inplace":
                hidden[:] = hidden + vec
            elif mode == "replace":
                new_hidden = hidden + vec
                blocks[layer].output = (new_hidden, *out[1:]) if is_tuple else new_hidden
            else:
                raise ValueError(f"unknown steering mode {mode!r}")
        return final_logits(model, head, site)


@cell("steering", family="*", backend="hf", realizations=[{"mode": "inplace"}])
def steering_hf(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
                scale_by="mean_norm"):
    direction = (head_direction(model, m.head(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)

    def build(rows):
        return _steer_and_read(
            m.blocks(model), model, m.head(model), layer=layer, direction=direction, alpha=alpha,
            mode=mode, site="head", scale_by=scale_by, masked=True)

    return be.run(model, units, build)


@cell("steering", family="*", backend="vllm_async", realizations=[{"mode": "inplace"}])
def steering_vllm(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
                  residual=None, scale_by="mean_norm"):
    # the stream the strength is relative to follows the family's vLLM denotation (fused on the
    # RMSNorm families, plain on GPT-2) unless the row says otherwise
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    direction = (checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)

    def build(rows):
        return _steer_and_read(
            m.blocks(model), model, m.head(model), layer=layer, direction=direction, alpha=alpha,
            mode=mode, site="engine", residual=residual, scale_by=scale_by)

    return be.run(model, units, build)

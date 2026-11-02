"""Generation-time steering: the steering write applied at every decode step of a greedy
generation, reading each step's next-token logits (`[new_tokens, B, vocab]`). The write is a
whole-tuple replacement; `bound` selects the iteration form, `tracer.iter[0:N]` (bounded) or
`tracer.iter[:]` (unbounded). `alpha=0` writes nothing."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import checkpoint_direction, checkpoint_of, engine_logits, head_direction
from .registry import cell
from .steering import _attention_mask, _resolve_token, _strength_scale


def _steer_step(blocks, model, *, layer, direction, alpha, residual="plain", scale_by="mean_norm",
                masked=False):
    """One decode step's steering write into blocks[layer]'s output. Runs inside the trace, once
    per step. `masked` reads the padded batch's attention mask for the per-unit mean norm (HF)."""
    with torch.no_grad():
        mask = _attention_mask(model) if masked else None         # before the model runs past it
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        stream = _resid(out, residual)
        scale = _strength_scale(stream, scale_by, mask)
        vec = direction.to(device=hidden.device, dtype=hidden.dtype)
        new_hidden = hidden + (alpha * scale) * vec
        blocks[layer].output = (new_hidden, *out[1:]) if is_tuple else new_hidden


def _check_bound(bound):
    if bound not in ("bounded", "unbounded"):
        raise ValueError(f"unknown iteration bound {bound!r} (expected 'bounded' or 'unbounded')")


@cell("gen_steering", family="*", backend="hf", realizations=[{"bound": "unbounded"}])
def gen_steering_hf(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0,
                    bound="bounded", new_tokens=8, scale_by="mean_norm"):
    _check_bound(bound)
    direction = (head_direction(model, m.head(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)

    def step(index):
        if alpha != 0:
            _steer_step(m.blocks(model), model, layer=layer, direction=direction, alpha=alpha,
                        scale_by=scale_by, masked=True)
        return m.head(model).output[:, -1, :]            # this step's next-token logits [B, vocab]

    return be.generate(model, units, step, new_tokens=new_tokens, bounded=(bound == "bounded"))


@cell("gen_steering", family="*", backend="vllm_async", realizations=[{"bound": "unbounded"}])
def gen_steering_vllm(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0,
                      bound="bounded", new_tokens=8, residual=None, scale_by="mean_norm"):
    _check_bound(bound)
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    direction = (checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)

    def step(index):
        if alpha != 0:
            _steer_step(m.blocks(model), model, layer=layer, direction=direction, alpha=alpha,
                        residual=residual, scale_by=scale_by)
        return engine_logits(model)[-1:, :]              # the step's last row, [1, vocab]

    return be.generate(model, units, step, new_tokens=new_tokens, bounded=(bound == "bounded"))

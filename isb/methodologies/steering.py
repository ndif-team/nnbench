"""Steering (activation addition): add a steering vector into a block's residual-stream output and
read the model's next-token logits. The direction is the target token's unit unembedding row, built
before the trace; `alpha` is relative to the stream's own per-token norm; `alpha=0` writes nothing.
`mode` selects the write form, in place into the live buffer or whole-tuple replacement. `layers`,
when given, steers each listed block in one forward and replaces `layer`; `alpha` is then the total
strength, split evenly, so each block gets `alpha / len(layers)` (a single `layer` gets `alpha`).

`scale_by` names the norm the strength multiplies. "mean_norm" and "token_norm" read the stream the
steered forward itself produces at that block, so each later block's norm includes the earlier
blocks' writes (alpha 6 over 27 blocks of Qwen2.5-7B drives the logits nonfinite).
"calibrated_mean_norm" adds the same fixed vector at every steered block:
`alpha / len(layers) * n * d`, with `n` the unsteered stream's mean per-token norm at the first
steered block, measured once per loaded engine by a plain read forward (`calibrated_strengths`) and
reused by every later call. The steered forward reads nothing, so a system that adds a fixed value
at a hook steers any number of blocks in one forward, and the vectors sum to `alpha * n * d` at
every number of blocks: the stream stays finite and rows that differ in block count add the same
total."""
from __future__ import annotations

import functools
import weakref

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


CALIBRATED = "calibrated_mean_norm"
_CALIBRATED_NORMS = weakref.WeakKeyDictionary()      # backend -> {layer: unsteered mean norm}


def calibrated_strengths(be, layers, alpha, units, measure):
    """The strength of each of `layers` under `scale_by="calibrated_mean_norm"`, the same for all:
    `alpha / len(layers)` times the unsteered stream's mean per-token norm at the first of them.
    `measure(prompts, layers)` returns such norms from one plain forward of the system's own read;
    it runs on the first prompt of the first call that needs a layer's norm, and the value is kept
    for the loaded engine, so every later call (the harness's timed ones among them) adds fixed
    vectors and reads nothing."""
    first = min(layers)
    known = _CALIBRATED_NORMS.setdefault(be, {})
    if first not in known:
        (known[first],) = measure(units[:1], [first])
    return [alpha / len(layers) * known[first]] * len(layers)


def _mean_norms(be, model, blocks, prompts, layers, *, residual, masked):
    """The unsteered stream's mean per-token norm at each of `layers`, averaged over `prompts`: one
    read-only forward through `be.run`, `[len(layers)]` floats."""
    def build(rows):
        with torch.no_grad():
            mask = _attention_mask(model) if masked else None     # before the model runs past it
            return torch.stack([_strength_scale(_resid(blocks[i].output, residual), "mean_norm",
                                                mask).reshape(-1, 1)
                                for i in layers])                          # [L, B, 1]

    return be.run(model, prompts, build).mean(dim=(1, 2)).tolist()


def _strength_scale(stream, scale_by, mask=None):
    """What `alpha` multiplies, per unit: the stream's per-token norm averaged over the unit's real
    tokens ("mean_norm"; `mask` marks them in a left-padded batch, its rightmost columns being the
    stream's positions) or each token's own norm ("token_norm"). Shaped to broadcast over the
    stream: `[B, 1, 1]` for `[B, S, d]`, `[1, 1]` for vLLM's flat `[S, d]`."""
    norms = stream.norm(dim=-1)
    if scale_by == "token_norm":
        return norms.unsqueeze(-1)
    if scale_by != "mean_norm":
        raise ValueError(f"unknown steering scale {scale_by!r} (expected 'mean_norm', "
                         f"'token_norm' or {CALIBRATED!r})")
    if mask is None:
        return norms.mean(dim=-1, keepdim=True).unsqueeze(-1)
    m = mask[..., mask.shape[-1] - norms.shape[-1]:].to(norms.device, norms.dtype)
    return ((norms * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True)).unsqueeze(-1)


def _steer_and_read(blocks, model, head, *, layer=None, direction, alpha, mode, site,
                    residual="plain", scale_by="mean_norm", masked=False, layers=None,
                    strengths=None):
    """Steer blocks[layer], or each of blocks[layers] in forward order, along `direction` and read
    the next-token logits at `site`. Runs inside a trace. `strengths` (one float per steered block,
    from `calibrated_strengths`) adds those fixed multiples of the direction; without it each block
    gets `alpha / len(layers)` times the live norm `scale_by` names. `masked` reads the padded
    batch's attention mask for that per-unit mean norm (HF). The direction moves onto a block's
    device and dtype once, and again only for a block that sits elsewhere (an HF model split over
    GPUs)."""
    n = len(blocks)
    layers = [layer] if layers is None else list(layers)
    final = [i for i in layers if i % n == n - 1]
    if alpha != 0 and final:
        raise ValueError(
            f"steer layer {final[0]} is the read-out (final) block; pick layers < {n - 1} so the "
            f"write propagates through at least one block before the readout")
    with torch.no_grad():
        if alpha != 0:
            live = strengths is None
            mask = _attention_mask(model) if masked and live else None   # before the blocks run
            share = alpha / len(layers)
            moved = direction
            for k, layer in enumerate(layers):
                out = blocks[layer].output
                is_tuple = isinstance(out, tuple)
                hidden = out[0] if is_tuple else out
                if moved.device != hidden.device or moved.dtype != hidden.dtype:
                    moved = direction.to(device=hidden.device, dtype=hidden.dtype)
                if live:
                    vec = (share * _strength_scale(_resid(out, residual), scale_by, mask)) * moved
                else:
                    vec = strengths[k] * moved
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
                scale_by="mean_norm", layers=None):
    direction = (head_direction(model, m.head(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)
    strengths = None
    if scale_by == CALIBRATED and alpha != 0:
        measure = functools.partial(_mean_norms, be, model, m.blocks(model), residual="plain",
                                    masked=True)
        strengths = calibrated_strengths(be, [layer] if layers is None else list(layers), alpha,
                                         units, measure)

    def build(rows):
        return _steer_and_read(
            m.blocks(model), model, m.head(model), layer=layer, layers=layers, direction=direction,
            alpha=alpha, mode=mode, site="head", scale_by=scale_by, masked=True,
            strengths=strengths)

    return be.run(model, units, build)


@cell("steering", family="*", backend="vllm_async", realizations=[{"mode": "inplace"}])
def steering_vllm(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
                  residual=None, scale_by="mean_norm", layers=None):
    # the stream the strength is relative to follows the family's vLLM denotation (fused on the
    # RMSNorm families, plain on GPT-2) unless the row says otherwise
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    direction = (checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
                 if alpha != 0 else None)
    strengths = None
    if scale_by == CALIBRATED and alpha != 0:
        measure = functools.partial(_mean_norms, be, model, m.blocks(model), residual=residual,
                                    masked=False)
        strengths = calibrated_strengths(be, [layer] if layers is None else list(layers), alpha,
                                         units, measure)

    def build(rows):
        return _steer_and_read(
            m.blocks(model), model, m.head(model), layer=layer, layers=layers, direction=direction,
            alpha=alpha, mode=mode, site="engine", residual=residual, scale_by=scale_by,
            strengths=strengths)

    return be.run(model, units, build)

"""Generation-time cross-prompt patching: the transplant of activation patching injected at the
prefill step of a greedy generation of the corrupted prompt, returning the tokens the decode
produced (`[new_tokens, B, 1]`); a check call returns each step's next-token logits instead
(`[new_tokens, B, vocab]`). The injection lands on the first generation forward, which holds the
whole prompt; decode steps carry it through the request's own KV cache. `patch=False` is the
unpatched generation (the baseline). `bound` selects the iteration form."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from ..oracle.checks import generated_tokens_match, timed_check_of
from .readout import final_logits
from .registry import cell

timed_check_of("gen_patching")(generated_tokens_match)


def _capture(blocks, layer, residual):
    """The clean residual stream at `layer`, an independent snapshot. Runs inside trace 1."""
    with torch.no_grad():
        return _resid(blocks[layer].output, residual).clone()


def _inject_transplant(blocks, layer, clean_act, residual="plain"):
    """Replace `layer`'s stream with the clean snapshot on the prefill forward. Raises when the
    shapes differ: the pair is not length-matched or the prefill was chunked, so a position-aligned
    replacement is ill-defined."""
    with torch.no_grad():
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        clean = clean_act.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(clean.shape) != tuple(hidden.shape):
            raise ValueError(
                f"transplant shape {tuple(clean.shape)} != prefill residual {tuple(hidden.shape)}; "
                f"clean/corrupted prompts must be length-matched for position-aligned patching")
        if residual == "fused":
            if not (is_tuple and len(out) >= 2):
                raise ValueError("fused patch needs a (hidden, residual) layer output")
            blocks[layer].output = (clean - out[1], *out[1:])
        else:
            blocks[layer].output = (clean, *out[1:]) if is_tuple else clean


def _check_bound(bound):
    if bound not in ("bounded", "unbounded"):
        raise ValueError(f"unknown iteration bound {bound!r} (expected 'bounded' or 'unbounded')")


def _gen_patch_cell(be, model, m, units, *, layer, residual, bound, new_tokens, patch, read,
                    check):
    """A timed call decodes with no per-step work and returns the tokens; a check call returns
    `read()` at every step."""
    _check_bound(bound)
    clean = [unit[0] for unit in units]
    corrupted = [unit[1] for unit in units]
    h = m.blocks(model)
    bounded = bound == "bounded"

    def step(index):
        return read()

    per_step = step if check else None
    if not patch:
        return be.generate(model, corrupted, per_step, new_tokens=new_tokens, bounded=bounded,
                           keep_dtype=not check, tokens=not check)

    def capture():
        return _capture(h, layer, residual)

    def inject(clean_act):
        _inject_transplant(h, layer, clean_act, residual)

    return be.generate_patch(model, clean, corrupted, capture, inject, per_step,
                             new_tokens=new_tokens, bounded=bounded, keep_dtype=not check,
                             tokens=not check)


@cell("gen_patching", family="*", backend="hf", realizations=[{"bound": "unbounded"}])
def gen_patching_hf(be, model, m, units, *, layer=9, residual="plain", bound="bounded",
                    new_tokens=8, patch=True, check=False):
    head = m.head(model)

    def read():
        return final_logits(model, head, "head")

    return _gen_patch_cell(be, model, m, units, layer=layer, residual=residual, bound=bound,
                           new_tokens=new_tokens, patch=patch, read=read, check=check)


@cell("gen_patching", family="*", backend="vllm_async", realizations=[{"bound": "unbounded"}, {"residual": "plain"}])
def gen_patching_vllm(be, model, m, units, *, layer=9, residual=None, bound="bounded",
                      new_tokens=8, patch=True, check=False):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def read():
        return final_logits(model, None, "engine")

    return _gen_patch_cell(be, model, m, units, layer=layer, residual=residual, bound=bound,
                           new_tokens=new_tokens, patch=patch, read=read, check=check)

"""Generation-time steering — the WRITE × iteration composition (catalog roadmap item 1).

Steering (activation addition) applied at EVERY decode step of a greedy multi-token generation,
reading the per-step next-token logits. This is the serving-shaped workload: the intervention has
to survive the engine's decode loop, not just one forward. It is also the first cell measuring a
COMPOSITION of two separately-measured rows — replacement WRITE (in-place writes raise on vLLM,
replacement works) inside the iteration construct (unbounded tracer.iter[:] drops all per-step
saves on vLLM, bounded iter[0:N] works) — i.e. the "statuses compose upward" claim (design.md
§3.6) at method tier.
Externally it is the footprint of causalab's path_steering analysis, flagged "composition
unmeasured" in `docs/causalab-portability-audit.md` §4.

The realization axis here is the iteration BOUND (Level 1.5), not the write form (in-place vs
replace is already measured by the steering methodology — this cell writes replacement-only):

  - `bound="bounded"`   -> `tracer.iter[0:N]` — carries its own stop; the working idiom on vLLM.
  - `bound="unbounded"` -> `tracer.iter[:]`   — the documented idiom; works on HF (stop bound
    from max_new_tokens), drops ALL per-step saves on vLLM -> the frontier marker /
    flip-detector for the upstream fix.

Observable = per-step last-token logits, stacked `[new_tokens, vocab]`: HF reads
`lm_head.output[:, -1, :]` per step; vLLM reads the engine site `model.logits` (measured equal to
the portable unembed; Level-1 sites are portable on vLLM). Greedy on both backends, so per-step rows are comparable — a
near-tie token flip mid-generation diverges every later step, which is exactly the regime
sensitivity this workload exists to expose (the fp32 control separates precision from mechanism).

`alpha=0` performs no write at all — the unsteered baseline, same trace shape.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import cell
from .steering import _resolve_token


def _steer_step(blocks, head, *, layer, token_id, alpha):
    """One decode step's steering write: replacement-add `alpha` (relative to the residual's own
    per-token norm) of the target token's unembed direction into blocks[layer]'s output. Runs
    INSIDE the trace, once per iteration step. Replacement-only (the vLLM working form; in-place
    writes raise on vLLM)."""
    with torch.no_grad():                       # aux compute on inference tensors needs no_grad
        direction = F.normalize(head.weight[token_id].float(), dim=0).to(head.weight.dtype)
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        scale = hidden.norm(dim=-1).mean()      # self-calibrating: alpha is relative strength
        new_hidden = hidden + (alpha * scale) * direction
        blocks[layer].output = (new_hidden, *out[1:]) if is_tuple else new_hidden


def _check_bound(bound):
    if bound not in ("bounded", "unbounded"):
        raise ValueError(f"unknown iteration bound {bound!r} (expected 'bounded' or 'unbounded')")


@cell("gen_steering", family="gpt2", backend="hf")
def gen_steering_gpt2_hf(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0,
                         bound="bounded", new_tokens=8):
    _check_bound(bound)
    token_id = _resolve_token(model.tokenizer, target)

    def step():
        if alpha != 0:
            _steer_step(model.transformer.h, model.lm_head,
                        layer=layer, token_id=token_id, alpha=alpha)
        return model.lm_head.output[:, -1, :]            # this step's next-token logits [1, vocab]

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))


@cell("gen_steering", family="gpt2", backend="vllm_async")
def gen_steering_gpt2_vllm(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0,
                           bound="bounded", new_tokens=8):
    _check_bound(bound)
    token_id = _resolve_token(model.tokenizer, target)

    def step():
        if alpha != 0:
            _steer_step(model.transformer.h, model.lm_head,
                        layer=layer, token_id=token_id, alpha=alpha)
        # engine site == portable unembed (Level-1 sites are portable on vLLM); [-1:, :] keeps the step's LAST row so the
        # prefill step (which may carry more than one logits row) matches HF's [:, -1, :] read
        return model.logits[-1:, :]

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))

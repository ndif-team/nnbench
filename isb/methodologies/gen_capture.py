"""Per-step activation capture: the residual stream copied to host memory at every decode step.

The read of activation_harvest (same sites and denotations, see activation_harvest.py) placed inside
the generation loop. Each step reads the newest position: the prefill's last prompt token at step 0,
then each generated token. HF serves `[1, seq, d]` at the prefill and `[1, 1, d]` per decode step;
vLLM serves `[tokens, d]` for the request, every prompt token at the prefill and one row per decode
step (docs/models/vllm.md, "What your block sees on vLLM"). `be.last` takes the newest row on either,
so every step yields `[1, d]` and the run stacks `[new_tokens, d]`. The row is copied with `.cpu()`
inside the step, as an online monitor of a generation does; on vLLM the copy also keeps the read
intact, since a served value is the engine's live buffer. HF needs no `torch.no_grad()` here:
transformers' `GenerationMixin.generate` already runs under it.

The realization axis is the iteration bound, as for gen_steering: `bounded` (`tracer.iter[0:N]`,
the working idiom on vLLM) by default, `unbounded` (`tracer.iter[:]`) the declared alternative.
`export=False` is the baseline: the same decode loop, returning each step's greedy token.
"""
from __future__ import annotations

from ..profiles import _resid
from .gen_steering import _check_bound
from .observations import record_resolved
from .registry import cell


# alternative: the documented unbounded iter[:]
@cell("gen_capture", family="*", backend="hf", realizations=[{"bound": "unbounded"}])
def gen_capture_hf(be, model, m, prompts, *, layer, new_tokens, export=True, bound="bounded",
                   residual="plain"):
    _check_bound(bound)

    def step():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        if not export:
            return be.last(m.head(model).output).argmax(-1).cpu()           # this step's token [1]
        return be.last(_resid(m.blocks(model)[layer].output, residual)).cpu()   # newest row [1, d]

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))


@cell("gen_capture", family="*", backend="vllm_async", realizations=[{"bound": "unbounded"}])
def gen_capture_vllm(be, model, m, prompts, *, layer, new_tokens, export=True, bound="bounded",
                     residual=None):
    _check_bound(bound)
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def step():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        if not export:
            return model.logits[-1:, :].argmax(-1).cpu()
        return be.last(_resid(m.blocks(model)[layer].output, residual)).cpu()

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))

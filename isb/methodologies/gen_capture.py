"""Per-step activation capture: the residual stream's newest row copied to host memory at every
decode step, `[new_tokens, B, d]`. HF serves `[B, seq, d]` at the prefill and `[B, 1, d]` per step;
vLLM serves `[tokens, d]` per request; `be.last` takes the newest row on either. The copy inside the
step is the online monitor's own form and, on vLLM, what keeps the read intact (a served value is the
engine's live buffer). `export=False` is the baseline: the same decode with no per-step work,
returning the tokens it produced. `bound` selects the iteration form. A timed call returns the rows in the engine's
dtype, a check call in fp32."""
from __future__ import annotations

from ..oracle.checks import read_matches, timed_check_of
from ..profiles import _resid
from .gen_steering import _check_bound
from .observations import record_resolved
from .registry import cell

timed_check_of("gen_capture")(read_matches)


@cell("gen_capture", family="*", backend="hf", realizations=[{"bound": "unbounded"}])
def gen_capture_hf(be, model, m, units, *, layer, new_tokens, export=True, bound="bounded",
                   residual="plain", check=False):
    _check_bound(bound)

    def step(index):
        return be.last(_resid(m.blocks(model)[layer].output, residual)).cpu()      # [B, d]

    return be.generate(model, units, step if export else None, new_tokens=new_tokens,
                       bounded=(bound == "bounded"), keep_dtype=not check, tokens=not export)


@cell("gen_capture", family="*", backend="vllm_async", realizations=[{"bound": "unbounded"}])
def gen_capture_vllm(be, model, m, units, *, layer, new_tokens, export=True, bound="bounded",
                     residual=None, check=False):
    _check_bound(bound)
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def step(index):
        return be.last(_resid(m.blocks(model)[layer].output, residual)).cpu()      # [1, d]

    return be.generate(model, units, step if export else None, new_tokens=new_tokens,
                       bounded=(bound == "bounded"), keep_dtype=not check, tokens=not export)

"""Cell registry (design.md §12.1, §12.4, §12.8).

A cell is a fixed, explicit function for one (methodology, family, backend). It is keyed
by exactly those three; *variances* (prompts, which layers/sites to observe, idiomatic vs
portable formulation) are runtime PARAMS the cell accepts, not separate registrations.

    @cell("logit_lens", family="gpt2", backend="hf")
    def _(be, model, prompts, *, layers="all", unembed="module"): ...

§12.8 adds the family-generic form: a cell registered `family="*"` serves ANY family that
has a ModelProfile, receiving the profile as its third argument:

    @cell("logit_lens", family="*", backend="hf")
    def _(be, model, m, prompts, *, layers="all", unembed="module"): ...

`get_cell` binds the profile at lookup (partial application), so callers see the same
`fn(be, model, prompts, **params)` surface either way. An exact per-family registration
always wins over the generic one: that is the override hatch for a family whose
intervention code genuinely differs (the case worth keeping explicit).
"""
from __future__ import annotations

import functools
import inspect

from ..profiles import PROFILES

CELLS = {}  # (methodology, family, backend) -> fn(be, model, prompts, **params)


def cell(methodology: str, family: str, backend: str):
    def deco(fn):
        CELLS[(methodology, family, backend)] = fn
        return fn

    return deco


def get_cell(methodology: str, family: str, backend: str):
    def lookup(fam):
        fn = CELLS.get((methodology, fam, backend))
        if fn is None and backend.startswith("vllm_") and backend != "vllm_async":
            # Every vLLM *variant* runs the SAME vLLM model via the SAME intervention code as the
            # in-process async backend: the only difference is fully contained in the backend/model
            # object passed as `be` (over-HTTP for `vllm_serve`, an in-process sync engine for
            # `vllm_sync`, a pipeline/tensor-parallel engine for `vllm_pp`). So a variant cell is, by
            # construction, the `vllm_async` cell; reuse it rather than duplicating every
            # methodology×family. An explicit registration still takes precedence if a cell ever needs
            # to differ, and any silent divergence would surface as an oracle/test failure, not a
            # stale copy.
            fn = CELLS.get((methodology, fam, "vllm_async"))
        return fn

    fn = lookup(family)
    if fn is None:
        generic, profile = lookup("*"), PROFILES.get(family)
        if generic is not None and profile is not None:
            view = profile.for_backend(backend)         # engine view (e.g. a vLLM wrapper prefix)

            @functools.wraps(generic)                   # callers can reach the shared underlying fn
            def bound(be, model, prompts, *args, **params):    # bind the family's profile as `m`
                return generic(be, model, view, prompts, *args, **params)
            signature = inspect.signature(generic)
            bound.__signature__ = signature.replace(
                parameters=[p for i, p in enumerate(signature.parameters.values()) if i != 2])
            fn = bound
    return fn


def families_for(methodology: str, backend: str = "hf") -> list:
    fams = {f for (m, f, b) in CELLS if m == methodology and b == backend}
    if "*" in fams:                                     # a generic cell serves every profiled family
        fams.discard("*")
        fams.update(PROFILES)
    return sorted(fams)

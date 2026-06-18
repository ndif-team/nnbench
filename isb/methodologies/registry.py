"""Cell registry (design.md §12.1, §12.4).

A cell is a fixed, explicit function for one (methodology, family, backend). It is keyed
by exactly those three; *variances* (prompts, which layers/sites to observe, idiomatic vs
portable formulation) are runtime PARAMS the cell accepts — not separate registrations.

    @cell("logit_lens", family="gpt2", backend="hf")
    def _(be, model, prompts, *, layers="all", unembed="module"): ...
"""
from __future__ import annotations

CELLS = {}  # (methodology, family, backend) -> fn(be, model, prompts, **params)


def cell(methodology: str, family: str, backend: str):
    def deco(fn):
        CELLS[(methodology, family, backend)] = fn
        return fn

    return deco


def get_cell(methodology: str, family: str, backend: str):
    fn = CELLS.get((methodology, family, backend))
    if fn is None and backend.startswith("vllm_") and backend != "vllm_async":
        # Every vLLM *variant* runs the SAME vLLM model via the SAME intervention code as the
        # in-process async backend — the only difference is fully contained in the backend/model
        # object passed as `be` (over-HTTP for `vllm_serve`, an in-process sync engine for
        # `vllm_sync`, a pipeline/tensor-parallel engine for `vllm_pp`). So a variant cell is, by
        # construction, the `vllm_async` cell; reuse it rather than duplicating every
        # methodology×family. An explicit registration still takes precedence if a cell ever needs
        # to differ — and any silent divergence would surface as an oracle/test failure, not a
        # stale copy.
        fn = CELLS.get((methodology, family, "vllm_async"))
    return fn


def families_for(methodology: str, backend: str = "hf") -> list:
    return sorted(f for (m, f, b) in CELLS if m == methodology and b == backend)

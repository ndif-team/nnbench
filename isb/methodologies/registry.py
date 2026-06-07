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
    return CELLS.get((methodology, family, backend))


def families_for(methodology: str, backend: str = "hf") -> list:
    return sorted(f for (m, f, b) in CELLS if m == methodology and b == backend)

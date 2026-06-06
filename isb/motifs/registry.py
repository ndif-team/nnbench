"""Motif registry (design.md §11.7). A motif is CODE; it is family-independent and
consumes resolved Bindings, never paths. Each motif declares its primitive `requires`
set, which the runner feeds to `predict()` (§11.6, layering per §11.10).
"""
from __future__ import annotations

MOTIFS = {}
REQUIRES = {}


def motif(name: str, requires=frozenset()):
    def deco(fn):
        MOTIFS[name] = fn
        REQUIRES[name] = set(requires)
        return fn

    return deco


def build(name: str, workload, resolver):
    if name not in MOTIFS:
        raise KeyError(f"unknown motif {name!r}; registered: {sorted(MOTIFS)}")
    return MOTIFS[name](workload, resolver)


def requires_for(name: str) -> set:
    return REQUIRES.get(name, frozenset())

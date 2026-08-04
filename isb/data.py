"""Dataset registry (design.md §12.10): data is a replaceable feed, not part of a spec.

A spec describes a PROCEDURE (method, model requirements, task variants, expectations); the data
it runs over is a named, swappable source resolved here. One J-lens procedure runs any of the six
upstream eval distributions; the binding is an invocation choice (`--data`), with each spec
declaring a default so the standard sweep is unchanged.

A source yields UNITS: prompt strings, or (clean, corrupted) pairs — the driver already treats
units polymorphically. A source also carries KNOBS: per-dataset cell-param defaults (e.g. the
upstream readout-position rule), injected under any explicit task params. Unit kind is declared,
so binding pair-data to a prompt-procedure fails loudly instead of feeding tuples into a trace.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_JLENS_DIR = Path(__file__).resolve().parents[1] / "data" / "jlens"

# the upstream per-dataset readout rules (data/jlens/README-upstream.md); single copy
POSITION_RULES = {
    "lens-eval-association": "last",
    "lens-eval-multihop": "last",
    "lens-eval-multilingual": "last",
    "lens-eval-order-ops": "last",
    "lens-eval-poetry": "last_newline",
    "lens-eval-typo": "last",
}


@dataclass(frozen=True)
class DataRef:
    """A named binding to a data source: `name` picks the source, `n` optionally sizes it
    (generated sources require it; file sources default to all items)."""
    name: str
    n: int | None = None


@dataclass(frozen=True)
class Source:
    unit: str                    # "prompt" | "pair"
    load: object                 # callable(n | None) -> list of units
    knobs: dict = field(default_factory=dict)   # per-dataset cell-param defaults


def _jlens_loader(slug: str):
    def load(n=None):
        items = json.load(open(_JLENS_DIR / f"{slug}.json"))["items"]
        prompts = [it["prompt"] for it in items]
        return prompts[:n] if n else prompts
    return load


def _generated(fn):
    def load(n=None):
        if n is None:
            raise ValueError("generated sources need an explicit size (DataRef(name, n=...))")
        return fn(n)
    return load


def _sources() -> dict:
    from .tracegen import factual, few_shot_icl, ioi_pairs

    out = {
        "factual": Source("prompt", _generated(factual)),
        "ioi_pairs": Source("pair", _generated(ioi_pairs)),
        "few_shot_icl": Source("prompt", _generated(few_shot_icl)),
    }
    for slug, rule in POSITION_RULES.items():
        short = slug.removeprefix("lens-eval-")
        out[f"jlens/{short}"] = Source("prompt", _jlens_loader(slug), knobs={"position": rule})
    return out


SOURCES = _sources()


def load_data(ref: DataRef) -> tuple[list, dict]:
    """Resolve a DataRef -> (units, knobs). Unknown names are loud, with the available list."""
    if ref.name not in SOURCES:
        raise KeyError(f"unknown data source {ref.name!r}; available: {sorted(SOURCES)}")
    src = SOURCES[ref.name]
    return src.load(ref.n), dict(src.knobs)


def unit_kind(name: str) -> str:
    return SOURCES[name].unit

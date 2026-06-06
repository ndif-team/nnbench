"""Runner — the verification flow (design.md §11.8).

`run_cell` executes one (workload × backend) cell, failure-tolerantly, and returns the
raw outcome. `evaluate` then turns a set of cells into applicability states by comparing
each against the HF reference via the oracle — assigning SUPPORTED / SILENTLY_WRONG /
ERROR / UNSUPPORTED_BY_CONSTRUCTION (§8.1).
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from ..motifs.registry import build as build_motif
from ..motifs.registry import requires_for
from ..oracle.equivalence import compare, is_equivalent
from ..resolve import Resolver, predict
from ..spec.schema import AppState


@dataclass
class CellResult:
    backend: str
    predicted: str
    state: str
    latency_s: Optional[float] = None
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    value: Any = None       # cpu tensor; cleared by evaluate() once compared


def run_cell(
    workload,
    repo: str,
    family,
    backend_profile,
    backend_impl,
    *,
    run_negatives: bool = True,
    timeout_s: float = 600.0,
) -> CellResult:
    predicted = predict(
        workload, family, backend_profile, requires_for(workload.motif)
    )
    if predicted == AppState.UNSUPPORTED_BY_CONSTRUCTION and not run_negatives:
        return CellResult(backend_profile.name, predicted, predicted)

    model = None
    try:
        model = backend_impl.load(repo)
        resolver = Resolver(family, model)
        program = build_motif(workload.motif, workload, resolver)
        prompt = workload.inputs.prompts[0]
        t0 = time.time()
        out = backend_impl.run(model, program, prompt, workload.generation)
        _ = time.time() - t0  # wall clock incl. trace overhead, run() reports its own
        return CellResult(
            backend_profile.name,
            predicted,
            "RAN",
            latency_s=out["latency_s"],
            value=out["value"],
        )
    except Exception as e:  # failure-tolerant: a cell never crashes the sweep
        traceback.print_exc()
        return CellResult(
            backend_profile.name, predicted, AppState.ERROR, error=repr(e)[:300]
        )
    finally:
        if model is not None:
            backend_impl.teardown(model)


def evaluate(cells, reference: str = "hf", top1_thresh: float = 0.8):
    """Assign applicability states by comparing each cell to the reference cell."""
    ref = next((c for c in cells if c.backend == reference), None)
    refval = ref.value if ref is not None else None
    for c in cells:
        if c.state in (AppState.ERROR, AppState.UNSUPPORTED_BY_CONSTRUCTION):
            continue
        if c.backend == reference:
            c.state = AppState.SUPPORTED            # reference is ground truth
        else:
            c.metrics = compare(refval, c.value)
            c.state = (
                AppState.SUPPORTED
                if is_equivalent(c.metrics, top1_thresh)
                else AppState.SILENTLY_WRONG
            )
    for c in cells:                                  # drop tensors after comparison
        c.value = None
    return cells

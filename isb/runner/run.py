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


def run_workloads_on_backend(
    workloads,
    repo: str,
    family,
    backend_profile,
    backend_impl,
    *,
    run_negatives: bool = True,
):
    """Load a backend ONCE, run each workload, tear down once.

    Returns {workload.id: CellResult}. Use ONLY for backends whose engine survives a
    failed intervention. F-3 (docs/findings.md) showed a vLLM worker error can kill the
    EngineCore, poisoning later workloads — so the sweep isolates vLLM cells with
    per-cell `run_cell` instead. This load-once path is for engine-stable backends (HF).
    """
    results = {}
    pending = []
    for wl in workloads:
        predicted = predict(wl, family, backend_profile, requires_for(wl.motif))
        if predicted == AppState.UNSUPPORTED_BY_CONSTRUCTION and not run_negatives:
            results[wl.id] = CellResult(backend_profile.name, predicted, predicted)
        else:
            pending.append((wl, predicted))
    if not pending:
        return results

    model = None
    try:
        model = backend_impl.load(repo)
        for wl, predicted in pending:
            try:
                resolver = Resolver(family, model)
                program = build_motif(wl.motif, wl, resolver)
                out = backend_impl.run(
                    model, program, wl.inputs.prompts[0], wl.generation
                )
                results[wl.id] = CellResult(
                    backend_profile.name, predicted, "RAN",
                    latency_s=out["latency_s"], value=out["value"],
                )
            except Exception as e:
                traceback.print_exc()
                results[wl.id] = CellResult(
                    backend_profile.name, predicted, AppState.ERROR, error=repr(e)[:300]
                )
    except Exception as e:  # load failed -> every pending cell is ERROR
        for wl, predicted in pending:
            results[wl.id] = CellResult(
                backend_profile.name, predicted, AppState.ERROR,
                error=f"backend load failed: {repr(e)[:200]}",
            )
    finally:
        if model is not None:
            backend_impl.teardown(model)
    return results


def evaluate(cells, reference: str = "hf", top1_thresh: float = 0.9, tv_tol: float = 0.05):
    """Assign applicability states by comparing each cell to the reference cell.

    If the reference cell itself failed (no value), a successfully-running non-reference
    cell is `NO_REFERENCE` — it ran, but there is no ground truth to judge it against
    (NOT SILENTLY_WRONG, which would falsely condemn a correct result).
    """
    ref = next((c for c in cells if c.backend == reference), None)
    refval = ref.value if ref is not None else None
    for c in cells:
        if c.state in (AppState.ERROR, AppState.UNSUPPORTED_BY_CONSTRUCTION):
            continue
        if c.backend == reference:
            c.state = AppState.SUPPORTED if refval is not None else AppState.NO_REFERENCE
        elif refval is None:
            c.state = AppState.NO_REFERENCE
        else:
            c.metrics = compare(refval, c.value)
            c.state = (
                AppState.SUPPORTED
                if is_equivalent(c.metrics, top1_thresh, tv_tol)
                else AppState.SILENTLY_WRONG
            )
    for c in cells:                                  # drop tensors after comparison
        c.value = None
    return cells

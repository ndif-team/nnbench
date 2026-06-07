"""Runner (design.md §11.8 flow, §12.2 per-family control).

`run_cell` runs one cell `(methodology, family, backend)` with given prompts+params,
failure-tolerantly. `evaluate` scores a set of cells that share a (methodology, family,
params) task across backends, using **HF as the per-family control** (§12.2): each
non-HF cell is compared to the HF cell via the oracle.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from ..methodologies.registry import get_cell
from ..oracle.equivalence import compare, is_equivalent
from ..states import AppState


@dataclass
class PerfResult:
    """Performance measurement for one warm cell run (design.md §8.2). Filled by the perf path
    (warmup + N trials), never by the cheap single-shot correctness path."""
    median_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    n_trials: int
    warmup: int
    peak_mem_mb: float
    throughput: Optional[float] = None            # prompts/s (batched); None for interactive single
    overhead_vs_baseline: Optional[float] = None  # median / no-intervention-baseline median (ratio)
    enforce_eager: Optional[bool] = None          # vLLM: True (CUDA graphs forced off) — fairness note
    notes: str = ""


@dataclass
class CellResult:
    methodology: str
    family: str
    backend: str
    label: str                       # human label for the variant/params (display only)
    state: str
    latency_s: Optional[float] = None
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)
    value: Any = None                # cpu tensor; cleared by evaluate() after comparison
    workload: str = "interactive"    # "interactive" | "batched" — a coverage axis (oracle-checked per regime)
    perf: Optional["PerfResult"] = None  # filled only by the perf path, only for SUPPORTED*/cells


def run_cell(
    methodology, family, backend_name, backend_impl, repo, prompts,
    *, params=None, label="",
) -> CellResult:
    params = params or {}
    fn = get_cell(methodology, family, backend_name)
    if fn is None:
        return CellResult(
            methodology, family, backend_name, label, AppState.UNSUPPORTED,
            error="no cell for this combination",
        )
    model = None
    try:
        model = backend_impl.load(repo)
        t0 = time.time()
        value = fn(backend_impl, model, prompts, **params)
        return CellResult(
            methodology, family, backend_name, label, "RAN",
            latency_s=time.time() - t0, value=value,
        )
    except Exception as e:  # failure-tolerant: a cell never crashes the sweep
        traceback.print_exc()
        return CellResult(
            methodology, family, backend_name, label, AppState.ERROR, error=repr(e)[:300]
        )
    finally:
        if model is not None:
            backend_impl.teardown(model)


def evaluate(cells, control: str = "hf", top1_thresh: float = 0.9, tv_tol: float = 0.05):
    """Score cells against the per-(methodology, family) control (§12.2).

    Cells are grouped by (methodology, family) and each group is scored against ITS OWN
    control cell — so vLLM-Llama is compared to HF-Llama, never to HF-GPT2. A non-control
    cell that ran is SUPPORTED iff equivalent to its control, else SILENTLY_WRONG;
    NO_REFERENCE if its control failed.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for c in cells:
        # group by workload too: batching is a regime that can change correctness, so HF-batched is
        # the control for vLLM-batched, never HF-interactive.
        groups[(c.methodology, c.family, c.workload)].append(c)

    for group in groups.values():
        ctrl = next((c for c in group if c.backend == control), None)
        refval = ctrl.value if ctrl is not None else None
        for c in group:
            if c.state in (AppState.ERROR, AppState.UNSUPPORTED):
                continue
            if c.backend == control:
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
    for c in cells:
        c.value = None
    return cells


def disambiguate_precision(
    cells, control_value, rerun_at_control_dtype,
    control: str = "hf", top1_thresh: float = 0.9, tv_tol: float = 0.05,
):
    """Separate a precision degradation from a real bug for `SILENTLY_WRONG` non-control cells.

    The strict oracle flags ANY top1/TV gate failure as `SILENTLY_WRONG`, but a near-tie that only
    diverges because the backend ran a lower precision than the control is a DEGRADATION, not a bug
    (e.g. vLLM bf16 vs HF fp32 flipping a near-tie top-1). For each `SILENTLY_WRONG` non-control
    cell, re-run it at the control's precision via `rerun_at_control_dtype(cell) -> cpu tensor |
    None`; if that matches `control_value`, relabel the cell `SUPPORTED_DEGRADED`. Mutates
    `cell.state` / `cell.metrics`. Call AFTER `evaluate`. Returns `cells`.
    """
    if control_value is None:
        return cells
    for c in cells:
        if c.backend == control or c.state != AppState.SILENTLY_WRONG:
            continue
        ctrl_dtype_value = rerun_at_control_dtype(c)
        if ctrl_dtype_value is None:
            continue
        m = compare(control_value, ctrl_dtype_value)
        c.metrics["control_dtype_tv"] = round(m["tv"], 4)
        if is_equivalent(m, top1_thresh, tv_tol):
            c.state = AppState.SUPPORTED_DEGRADED  # matched at the control's precision -> precision, not a bug
    return cells

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
        groups[(c.methodology, c.family)].append(c)

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

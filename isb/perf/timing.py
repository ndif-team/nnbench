"""Self-contained timing helper for the perf path (design.md §8.2).

Mirrors the nnsight timing methodology (`tests/performance/profile/profiler_utils.py`): load once →
warm up → N trials → center±spread, CUDA-synced around every timed region. We copy ~40 lines so the
benchmark has no dependency on nnsight's `tests/` tree, and make two deliberate choices: report the
**median** (robust to a stray cold/GC trial) and **GPU peak memory** via `max_memory_allocated`.

`time_cell` returns BOTH a `TimingResult` and the last warm output, so the oracle can check
correctness on the very run the perf numbers came from — never on an unverified regime.
"""
from __future__ import annotations

import gc
import statistics
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable


def sync_cuda() -> None:
    """Block until queued GPU work is done (no-op without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def force_gc() -> None:
    for _ in range(3):
        gc.collect()


def reset_peak_mem() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def peak_mem_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


@dataclass
class TimingResult:
    median_ms: float
    std_ms: float
    min_ms: float
    n_trials: int
    warmup: int
    peak_mem_mb: float
    times_ms: list = field(default_factory=list)


def time_cell(fn: Callable[[], Any], *, warmup: int = 3, n_trials: int = 7):
    """Warm up `warmup` calls (discarded — they pay engine/trace/CUDA-graph init), then time
    `n_trials` calls of `fn`. Each call of `fn` is the timed unit and opens its own fresh trace
    (required: the vLLM async generator is single-shot). Returns `(TimingResult, last_warm_output)`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    for _ in range(warmup):
        fn()
        force_gc()
    reset_peak_mem()                       # measure peak over the TIMED trials, after warmup settles
    times, out = [], None
    for _ in range(n_trials):
        sync_cuda()
        t0 = perf_counter()
        out = fn()
        sync_cuda()
        times.append((perf_counter() - t0) * 1000.0)
        force_gc()
    return (
        TimingResult(
            median_ms=statistics.median(times),
            std_ms=statistics.pstdev(times) if len(times) > 1 else 0.0,
            min_ms=min(times),
            n_trials=n_trials,
            warmup=warmup,
            peak_mem_mb=peak_mem_mb(),
            times_ms=times,
        ),
        out,
    )

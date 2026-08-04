"""GPU-free stub system to self-test the runner (scripts/perf.py). NOT a real backend.

Returns deterministic synthetic metrics derived from the config so the orchestration, JSON
round-trip, subprocess plumbing, and timeout path can be exercised without a model. Selected by
system="_echo". The overhead math is unit-tested separately against hand-built rows.
"""
from __future__ import annotations

import time

from ..core import Config

# synthetic gen latency shape: baseline (op none) cheapest; read scales with footprint; steer cheap.
_FP = {"one": 1.0, "half": 3.0, "all": 6.0}
_OP = {"none": 0.0, "read": 1.0, "steer": 0.2, "qk": 1.5}


def run(cfg: Config) -> dict:
    if cfg.op == "hang":                      # lets a test exercise the timeout/HANG path
        time.sleep(cfg.timeout_s + 60)
    base = 0.010
    extra = base * _OP.get(cfg.op, 0.0) * _FP.get(cfg.footprint, 1.0)
    gen = base + extra
    return {
        "median_total_lat_s": gen, "std_total_lat_s": 0.0,
        "median_gen_lat_s": gen, "std_gen_lat_s": 0.0,
        "peak_mem_mb": 100.0 + extra * 1000.0,
        "artifact_kb": 0.0, "transfer_bytes": 0, "correct": None,
    }

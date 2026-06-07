"""Timing-helper tests (isb/perf/timing.py) — no GPU; pure logic with a controlled clock.

We monkeypatch `perf_counter` so the trial durations are deterministic, then assert: warmup calls
are discarded (not timed), the median/min/std come from the timed trials only, and the last warm
output is returned (so the oracle can check correctness on the timed run).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.perf.timing as timing  # noqa: E402


def _run_with_clock(fn, seq, **kw):
    it = iter(seq)
    orig = timing.perf_counter
    timing.perf_counter = lambda: next(it)
    try:
        return timing.time_cell(fn, **kw)
    finally:
        timing.perf_counter = orig


def test_warmup_discarded_and_stats():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]                       # output == cumulative call count

    # 3 timed trials -> two perf_counter reads each; durations 0.5s, 0.25s, 0.125s (exact in float)
    seq = [0.0, 0.5, 1.0, 1.25, 2.0, 2.125]
    res, out = _run_with_clock(fn, seq, warmup=2, n_trials=3)

    assert calls["n"] == 5                        # 2 warmup + 3 timed
    assert out == 5                               # last warm output, not a warmup one
    assert res.n_trials == 3 and res.warmup == 2
    assert res.times_ms == [500.0, 250.0, 125.0]  # warmup never timed
    assert res.median_ms == 250.0
    assert res.min_ms == 125.0
    assert res.std_ms > 0.0


def test_zero_warmup_times_every_call():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return calls["n"]

    seq = [0.0, 0.1, 1.0, 1.1]                     # 2 trials, 100ms each
    res, _ = _run_with_clock(fn, seq, warmup=0, n_trials=2)
    assert calls["n"] == 2
    assert res.warmup == 0
    assert abs(res.median_ms - 100.0) < 1e-6
    assert res.std_ms == 0.0 or res.std_ms < 1e-6  # both trials equal


def test_n_trials_must_be_positive():
    raised = False
    try:
        timing.time_cell(lambda: None, warmup=1, n_trials=0)
    except ValueError:
        raised = True
    assert raised


def test_sync_and_gc_are_noops_without_cuda():
    # both must be safe to call with no CUDA present
    timing.sync_cuda()
    timing.force_gc()
    assert timing.peak_mem_mb() == 0.0 or timing.peak_mem_mb() >= 0.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

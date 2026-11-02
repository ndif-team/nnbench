"""Phase-profiler split of a call that sends two requests (isb/perf/phase_profiler.py) — no GPU.

A synthetic call reads a stream in one request and writes with it in the next: each request's
engine window is charged on its own, the client time between them goes to the second request's
preparation, and the floor split counts the compiled floor once per request.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.perf.phase_profiler import _bounds, _decompose, split_with_floors  # noqa: E402

US = 1000                                        # boundaries are perf_counter ns


def test_two_request_call_splits_per_request():
    t0, t7 = 1000 * US, 2300 * US
    sent, received = [1100 * US, 1700 * US], [1500 * US, 2100 * US]
    events = [{"event": "arrival", "t": 1150 * US},
              {"event": "step", "start": 1160 * US, "end": 1450 * US, "finished": 1},
              {"event": "arrival", "t": 1760 * US},
              {"event": "step", "start": 1770 * US, "end": 1820 * US, "finished": 0},
              {"event": "step", "start": 1830 * US, "end": 2060 * US, "finished": 1}]
    marks = {1450 * US: [(1160 * US, "base"), (1200 * US, "hook"), (1220 * US, "base"),
                         (1440 * US, "between")],
             2060 * US: [(1770 * US, "base"), (1800 * US, "readout"), (1810 * US, "base"),
                         (1820 * US, "between"), (1830 * US, "base"), (2050 * US, "between")]}
    # a kernel launched inside the first request's hook region, one launched from another thread
    activities = [[1300 * US, 1400 * US, 1210 * US, True], [1900 * US, 1950 * US, 1840 * US, False]]

    trial = _bounds(t0, t7, sent, received, events)
    assert trial["requests"] == [[1100 * US, 1150 * US, 1450 * US, 1500 * US],
                                 [1700 * US, 1760 * US, 2060 * US, 2100 * US]]
    d = _decompose(trial, marks, activities)

    ms = 1e-3                                    # one boundary unit (US ns) in ms
    sub = d["subphases_ms"]
    assert sub["client_prep"] == pytest.approx((100 + 200) * ms)    # t0..t1, then t4..t1 between
    assert sub["request_transfer"] == pytest.approx((50 + 60) * ms)
    assert sub["engine"] == pytest.approx((300 + 300) * ms)
    assert sub["engine_hook"] == pytest.approx((20 + 100) * ms)     # its region and its kernel
    assert sub["engine_readout"] == pytest.approx(10 * ms)
    assert sub["output_transfer"] == pytest.approx((50 + 40) * ms)
    assert sub["client_post"] == pytest.approx(200 * ms)            # after the last request only
    seg = d["segments_raw_ms"]
    assert seg == pytest.approx({"engine_outside_spans": 470 * ms,
                                 "hook_dispatch_and_edit": (300 + 110 + 120) * ms,
                                 "readout_and_host_ipc": (10 + 90 + 200) * ms})
    assert sum(seg.values()) == pytest.approx(d["e2e_wall_ms"]) == 1300 * ms
    assert d["requests"] == 2 and d["boundaries_ordered"] and d["engine_steps"] == 3
    assert d["other_thread_kernel_ms"] == pytest.approx(50 * ms)

    def row(label, trials):
        return {"label": label, "units_sha256": "u", "new_tokens": 1, "trials": trials,
                "segments_raw_ms": {}, "segments_raw_spans_ms": {}, "subphases_ms": {},
                "alignment": {"bracket_width_ns": 900.0}, "conservation": {},
                **dict.fromkeys(("uninstrumented_e2e_ms", "uninstrumented_before_e2e_ms",
                                 "uninstrumented_after_e2e_ms", "uninstrumented_gap_fraction",
                                 "instrumentation_overhead_fraction"), 0.0)}

    floor = {"e2e_wall_ms": 0.25, "requests": 1, "marks": 4, "segments_raw_ms": {
        "engine_outside_spans": 0.2, "hook_dispatch_and_edit": 0.03, "readout_and_host_ipc": 0.02}}
    profiles = [{"backend": "vllm-plain/9-9-9", "case": "steer_probe/tiny@set", "gpus": 1,
                 "vllm_version": "9.9.9", "engine_options": {},
                 "rows": {"interactive|__vanilla__": row("__vanilla__", [floor])}},
                {"backend": "streamtap/patched", "case": "steer_probe/tiny@set", "gpus": 1,
                 "vllm_version": "9.9.9", "engine_options": {},
                 "rows": {"interactive|nudge": row("nudge", [d])}}]
    split = {r["backend"]: r for r in split_with_floors(profiles)}["streamtap/patched"]
    assert split["forwards_per_call"] == 2
    assert split["segments_ms"] == pytest.approx({
        "base_compiled_engine": 2 * 0.2, "eager_graph_break_tax": 470 * ms - 2 * 0.2,
        "hook_dispatch_and_edit": 530 * ms, "readout_and_host_ipc": 300 * ms})

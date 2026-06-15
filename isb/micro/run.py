"""Micro-tier runner: one backend per process, all its probes against one loaded
GPT-2, each probe in a watchdog thread.

HANG handling: a probe that exceeds its timeout is recorded HANG and the sweep for
that backend STOPS — the stuck thread cannot be killed and (on vLLM) still owns the
persistent event loop, so later probes would measure a poisoned engine, not the
primitive. Probe registration order is safest-first for exactly this reason.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..states import AppState
from .probes import PROBES, expected_state, names_for

PROBE_TIMEOUT_S = 180.0


@dataclass
class ProbeResult:
    name: str
    backend: str
    state: str
    note: str
    latency_s: float | None = None
    expected: str = AppState.SUPPORTED
    surprise: bool = False


def _run_one(fn, be, model, timeout_s: float):
    box = {}

    def target():
        try:
            box["result"] = fn(be, model)
        except Exception as e:  # clean per-probe ERROR (the runner records, never crashes)
            msg = str(e).strip().splitlines()
            box["error"] = msg[-1] if msg else type(e).__name__
        # a watchdog timeout leaves box empty -> HANG

    t = threading.Thread(target=target, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout_s)
    dt = time.perf_counter() - t0
    if t.is_alive():
        return AppState.HANG, f"exceeded {timeout_s:.0f}s watchdog", dt
    if "error" in box:
        return AppState.ERROR, box["error"], dt
    state, note = box["result"]
    return state, note, dt


def run_micro(backend_name: str, repo: str = "openai-community/gpt2",
              only: list | None = None, timeout_s: float = PROBE_TIMEOUT_S) -> list:
    from ..backends import IMPLS

    be = IMPLS[backend_name]()
    model = be.load(repo)
    results = []
    try:
        for name in names_for(backend_name):
            if only and name not in only:
                continue
            state, note, dt = _run_one(PROBES[(name, backend_name)], be, model, timeout_s)
            exp = expected_state(name, backend_name)
            surprise = state != exp
            results.append(ProbeResult(name, backend_name, state, note, dt, exp, surprise))
            flag = f"  <- SURPRISE (expected {exp})" if surprise else ""
            print(f"  {name:<20}{state:<18}{dt:6.1f}s  {note}{flag}", flush=True)
            if state == AppState.HANG:
                print("  -- HANG poisons the engine; aborting this backend's remaining probes",
                      flush=True)
                break
    finally:
        be.teardown(model)
    return results


def print_micro_map(backend_name: str, repo: str, results: list) -> None:
    print("\n=== Micro tier — Level 0/1 primitive map ===")
    print(f"backend: {backend_name}    model: {repo}")
    print("-" * 100)
    print(f"{'probe':<21}{'state':<18}{'expected':<18}{'!':<3}{'lat':<7}{'denotation check / note'}")
    print("-" * 100)
    for r in results:
        lat = f"{r.latency_s:.1f}s" if r.latency_s is not None else "-"
        mark = "*" if r.surprise else ""
        print(f"{r.name:<21}{r.state:<18}{r.expected:<18}{mark:<3}{lat:<7}{r.note}")
    print("-" * 100)
    surprises = [r for r in results if r.surprise]
    n_ok = sum(1 for r in results if r.state == AppState.SUPPORTED)
    print(f"{n_ok}/{len(results)} SUPPORTED; {len(surprises)} surprise(s) vs expected.")
    for r in surprises:
        print(f"  * {r.name}: expected {r.expected} -> got {r.state}  ({r.note})")

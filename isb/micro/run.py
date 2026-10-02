"""Micro-tier runner: one backend per process, all its probes against one loaded
GPT-2, each probe in a watchdog thread.

HANG handling: a probe that exceeds its timeout is recorded HANG and the sweep for
that backend STOPS — the stuck thread cannot be killed and (on vLLM) still owns the
persistent event loop, so later probes would measure a poisoned engine, not the
primitive. Probe registration order is safest-first for exactly this reason.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

from ..states import AppState
from .probes import PROBES, names_for

PROBE_TIMEOUT_S = 180.0


@dataclass
class ProbeResult:
    name: str
    backend: str
    state: str
    note: str
    latency_s: float | None = None
    detail: str | None = None


def _error_note(e: BaseException) -> tuple[str, str]:
    """(`Type: message`, full text) for a probe's exception. A vLLM worker error carries the
    remote traceback in its message, whose last line is a caret marker, so the note is the last
    line that reads like an exception (else the last message line), never a source excerpt."""
    text = str(e)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    msgs = [ln for ln in lines if not set(ln) <= set("^~ ") and not ln.startswith(("File ", "Traceback"))]
    errlike = [ln for ln in msgs if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\b", ln)]
    head = errlike[-1] if errlike else (msgs[-1] if msgs else "")
    name = type(e).__name__
    note = head if head.startswith(name) else (f"{name}: {head}" if head else name)
    return note, text[:2000]


def _run_one(fn, be, model, timeout_s: float):
    box = {}

    def target():
        try:
            box["result"] = fn(be, model)
        except Exception as e:  # clean per-probe ERROR (the runner records, never crashes)
            box["error"], box["detail"] = _error_note(e)
        # a watchdog timeout leaves box empty -> HANG

    t = threading.Thread(target=target, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout_s)
    dt = time.perf_counter() - t0
    if t.is_alive():
        return AppState.HANG, f"exceeded {timeout_s:.0f}s watchdog", dt, None
    if "error" in box:
        return AppState.ERROR, box["error"], dt, box.get("detail")
    state, note = box["result"]
    return state, note, dt, None


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
            state, note, dt, detail = _run_one(PROBES[(name, backend_name)], be, model, timeout_s)
            results.append(ProbeResult(name, backend_name, state, note, dt, detail))
            print(f"  {name:<20}{state:<18}{dt:6.1f}s  {note}", flush=True)
            if state == AppState.HANG:
                print("  -- HANG poisons the engine; aborting this backend's remaining probes",
                      flush=True)
                break
    finally:
        be.teardown(model)
    return results


def micro_run_file(backend_name: str, repo: str, results: list,
                   out_dir: str, run_name: str) -> str:
    """Write a construct-probe run as a self-contained run file (isb/runfile.py). Probes
    self-check, so the file carries their states directly under ("probe", <name>) meta keys;
    there is nothing for a baseline comparison to score. Coordinates say "constructs": in this
    project's vocabulary "micro" is the perf op-cost benchmark, and these probes surface as
    construct support on backend pages."""
    from ..runfile import save_run
    from ..runs import EngineConfig, RunConfig, resolve_provenance, run_coordinates

    engine = (EngineConfig("transformers") if backend_name == "hf"
              else EngineConfig("vllm", mode="sync" if backend_name == "vllm_sync" else "async"))
    prov = resolve_provenance(RunConfig(engine=engine))
    prov["coordinates"] = run_coordinates(
        spec="constructs", methodology="constructs", family="gpt2", repo=repo,
        interface=backend_name, cases=[{"label": r.name} for r in results])
    meta = {("probe", r.name): {"state": r.state, "note": r.note, "latency_s": r.latency_s,
                                "detail": r.detail}
            for r in results}
    return save_run(out_dir, run_name, {("__meta__",): meta}, prov)


def print_micro_map(backend_name: str, repo: str, results: list) -> None:
    print("\n=== Micro tier — Level 0/1 primitive map ===")
    print(f"backend: {backend_name}    model: {repo}")
    print("-" * 100)
    print(f"{'probe':<21}{'state':<18}{'lat':<7}{'denotation check / note'}")
    print("-" * 100)
    for r in results:
        lat = f"{r.latency_s:.1f}s" if r.latency_s is not None else "-"
        print(f"{r.name:<21}{r.state:<18}{lat:<7}{r.note}")
    print("-" * 100)
    n_ok = sum(1 for r in results if r.state == AppState.SUPPORTED)
    print(f"{n_ok}/{len(results)} SUPPORTED.")

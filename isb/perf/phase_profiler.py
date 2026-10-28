"""Phase profiler: where one cell call's wall time goes, measured inside one run (the latency
breakdown figure, paper docs/fig3-profiling-plan.md).

In-container entry: `python3 -m isb.perf.phase_profiler` with `ISB_PHASE_PROFILE=1`, the job
directory at /job and the output at /output (scripts/profile_phases.py launches it through the
backend's own compose.yml). It builds the backend exactly as `isb.jobs.entry` does, binds every
row of the job's spec the way the harness does (`isb.sweep.execute._bind_case`), and times each
row (or the labels listed in `ISB_PHASE_ROWS`, a JSON list) and the backend's plain request with
the harness's warm-up and trial counts. Each row runs twice in the same engine: first with every
wrapper below passing straight through (`OFF` present; the harness's timing loop), then
instrumented, so the record carries the instrumentation's own cost.

Host timestamps are `time.perf_counter_ns()` (CLOCK_MONOTONIC, one clock for every process in the
container), taken at six boundaries of each timed call:

  t0  the client starts the call                      t7  the call returns, CUDA synced
  t1  the client calls its engine-core client with the request (SyncMPClient.add_request,
      AsyncMPClient.add_request_async, InprocClient.add_request), so the request's encoding
      and send fall in request_transfer
  t2  the engine core takes the request in (EngineCore.add_request, in the engine process)
  t3  the engine step that finishes the request returns (EngineCore.step / step_with_batch_queue)
  t4  the client's output processor sees the finished output (OutputProcessor.process_outputs)

Inside the engine, host regions have a kind: `hook` while interpretability-system code runs
(every PyTorch forward hook or pre-hook registered after this module loads, which covers vLLM-Lens,
vLLM-Hook and other hook-based systems, plus nnsight's `Interleaver.handle`, its taps replay and its
per-request bookkeeping in `Requests`); `readout` while that code runs a lens projection (the final
norm, the logits processor or `F.linear`), a device-to-host copy (`Tensor.cpu`) or nnsight's
packing of saves for the return (`Requests.harvest`, `Requests.record_saves`); `base` for the rest
of a step; `between` outside steps. At every change of kind, and at every step's start and end,
the engine records the host time c_i and a CUDA event on the current stream. The event completes
at g_i, when the GPU has finished everything launched before it, and never before c_i; the two
clocks are aligned at the request's first step, where the GPU is idle (offset = max_i of
c_i - event time_i). From g_i to g_(i+1) the GPU either runs the kernels launched in region i or
waits for that region's host code to launch them, so each such stretch is charged to region i's
kind: host work that overlaps earlier GPU work is charged to nothing, and nothing is counted
twice. The stretches telescope from t2 to t3 (clipped to that window, kept in order).

Subphases (they telescope to e2e = t7 - t0):
  client_prep  = t1 - t0     request_transfer = t2 - t1     engine = t3 - t2
  output_transfer = t4 - t3  client_post = t7 - t4
  engine_hook, engine_readout = the stretches charged to hook / readout regions
  engine_outside_spans = engine - engine_hook - engine_readout

Raw per-run segments, event attribution (`segments_raw_ms`):
  engine_outside_spans   = the base model and the engine's own host work, together
  hook_dispatch_and_edit = client_prep + request_transfer + engine_hook
  readout_and_host_ipc   = engine_readout + output_transfer + client_post

One extra call per row runs with torch.profiler on in the engine process (one session from the
request's first step to the step that finishes it), outside the timed trials and followed by a
fresh warm-up. It labels every device activity with the kind of region its launching runtime call
ran in (`traced`: kernel time per kind, the timeline) for diagnostics, and feeds the span
attribution (`segments_raw_spans_ms`), kept for comparison; it counts host span time on top of
the GPU time it overlaps:
  base_gpu_kernels       = base kernel time
  engine_host            = engine - base - interp - hook_kernels - readout_kernels
  hook_dispatch_and_edit = client_prep + request_transfer + (interp - readout) + hook_kernels
  readout_and_host_ipc   = readout + readout_kernels + output_transfer + client_post
(interp and readout here are the host time inside those spans.)

`split_with_floors` turns raw segments into the figure's four with the compiled stock floor of
the same workload, vLLM version and GPU count (a `vllm-plain` config without enforce_eager,
profiled in the same chain over the same units). The floor's plain request gives F, its engine
time outside interpretability code (engine_outside_spans; the same attribution on both sides):
  base_compiled_engine  = F
  eager_graph_break_tax = engine time outside interpretability code - F
      (signed; it carries both a forced-eager engine and CUDA graphs run without torch.compile)
  hook_dispatch_and_edit, readout_and_host_ipc: as measured

Patches go in when this module is imported with ISB_PHASE_PROFILE set: in the client before the
engine starts, and in the engine process because vLLM starts it with fork (inherited) or spawn
(which re-imports this main module). Every wrapper returns straight to the wrapped function under
torch.compile tracing, so compiled graphs are unchanged, and while instrumentation is off; no
event is recorded while a stream is capturing a CUDA graph.
"""
from __future__ import annotations

import bisect
import functools
import json
import os
import queue
import random
import statistics
import threading
import time
import traceback
from pathlib import Path

import torch

FLAG = "ISB_PHASE_PROFILE"
LOG = Path("/dev/shm/isb_phase_profile.jsonl")      # engine process -> client, one line per event
TRACE = Path("/dev/shm/isb_phase_trace")            # present: the engine traces its steps
OFF = Path("/dev/shm/isb_phase_off")                # present: every wrapper passes through
_COUNTERS = ("interp_ns", "readout_ns")
_state = {"on": True, "interp": 0, "readout": 0, "in_step": False, "tracing": False,
          "prof": None, **dict.fromkeys(_COUNTERS, 0)}
_RANGES = {"isb_interp": "hook", "isb_readout": "readout"}       # host ranges of a traced call
_KINDS = ("base", "hook", "readout")
_SEGMENT = {"between": "engine", "base": "engine", "hook": "hook", "readout": "readout"}
# boundaries of the request in flight: host times with the kind that starts there, their events
_bounds = {"active": False, "marks": [], "events": [], "free": [], "queue": None}
_client = {"sent": [], "received": []}


def _compiling():
    return torch.compiler.is_compiling()


def _capturing():
    return torch.cuda.is_current_stream_capturing()


def _new_event():
    return torch.cuda.Event(enable_timing=True)


def _mark(kind):
    """A boundary of the request in flight: the host region that starts now is `kind`."""
    if not _bounds["active"] or _capturing():
        return
    free = _bounds["free"]
    event = free.pop() if free else _new_event()
    _bounds["marks"].append((time.perf_counter_ns(), kind))
    event.record()
    _bounds["events"].append(event)


def _read_bounds(work):
    """Engine-process thread: once a request's last event completes, log every boundary's host
    time and event time (ms after the first event), keyed by the finishing step's end."""
    while True:
        end, marks, events = work.get()
        for event in events:               # they may sit on different streams
            event.synchronize()
        _log({"event": "bounds", "end": end, "marks": marks,
              "gpu_ms": [events[0].elapsed_time(e) for e in events]})
        _bounds["free"].extend(events)


def _submit_bounds(end):
    """Hand the finished request's boundaries to the reader thread, off the engine's path."""
    if _bounds["queue"] is None:
        _bounds["queue"] = queue.Queue()
        threading.Thread(target=_read_bounds, args=(_bounds["queue"],), daemon=True).start()
    _bounds["queue"].put((end, _bounds["marks"], _bounds["events"]))
    _bounds.update(active=False, marks=[], events=[])


def _interp_span(fn):
    """Run `fn` as interpretability-system code; nested spans count once."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _compiling() or not _state["on"]:
            return fn(*args, **kwargs)
        outer = _state["interp"] == 0
        _state["interp"] += 1
        record = None
        if outer:
            _mark("hook")
            if _state["tracing"]:
                record = torch.profiler.record_function("isb_interp")
                record.__enter__()
        start = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            if outer:
                _state["interp_ns"] += time.perf_counter_ns() - start
                if record is not None:
                    record.__exit__(None, None, None)
                _mark("base" if _state["in_step"] else "between")
            _state["interp"] -= 1
    return wrapper


def _readout_span(fn):
    """Run `fn` as readout when it runs inside an interp span."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _compiling() or _state["interp"] == 0 or _state["readout"]:
            return fn(*args, **kwargs)
        _state["readout"] += 1
        _mark("readout")
        record = torch.profiler.record_function("isb_readout") if _state["tracing"] else None
        if record is not None:
            record.__enter__()
        start = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            _state["readout_ns"] += time.perf_counter_ns() - start
            if record is not None:
                record.__exit__(None, None, None)
            _mark("hook")
            _state["readout"] -= 1
    return wrapper


def _log(event):
    with LOG.open("a") as f:
        f.write(json.dumps(event) + "\n")


def _step_trace(prof):
    """A traced request's timeline in the profiler's clock (us), for diagnostics: the window from
    its first step's start to its finishing step's end, the steps, the interp and readout ranges
    on the host, and every device activity (kernel, copy, set) with the kind of range its
    launching runtime call ran in. A device activity shares its correlation id with that call;
    one without a matching call counts as base and is reported. A `record_function` range also
    appears on the device as a user annotation spanning its kernels; those are spans, not GPU
    work, and are left out."""
    from torch.autograd import DeviceType

    events = prof.events()
    host = [e for e in events if e.device_type == DeviceType.CPU]
    steps = [e.time_range for e in host if e.name == "isb_step"]
    ranges = sorted([e.time_range.start, e.time_range.end, _RANGES[e.name]]
                    for e in host if e.name in _RANGES)
    launches = {e.id: e.time_range.start for e in host if e.name.startswith("cu")}
    index = {kind: [r for r in ranges if r[2] == kind] for kind in ("readout", "hook")}
    kernels, unmatched = [], 0.0
    for e in events:
        if e.device_type != DeviceType.CUDA or e.is_user_annotation:
            continue
        launched = launches.get(e.id)
        if launched is None:
            unmatched += e.time_range.elapsed_us()
        kernels.append([e.time_range.start, e.time_range.end,
                        "base" if launched is None else _kind_at(index, launched)])
    return {"window": [min(r.start for r in steps), max(r.end for r in steps)],
            "steps": [[r.start, r.end] for r in steps], "ranges": ranges, "kernels": kernels,
            "unmatched_kernel_us": unmatched}


def _kind_at(index, t):
    """The kind of host range open at `t` (`index`: each kind's ranges sorted by start): readout,
    then hook, else base. Ranges of one kind do not overlap (only the outermost interp span and
    the outermost readout span open one)."""
    for kind in ("readout", "hook"):
        own = index[kind]
        i = bisect.bisect_right(own, [t, float("inf")]) - 1
        if i >= 0 and t <= own[i][1]:
            return kind
    return "base"


def _step(fn):
    """An engine step: its wall time, spans and boundaries. Reads `OFF` once per step, which
    switches every wrapper in this process. The first step of a request opens its boundaries,
    and the step that finishes it hands them to the reader thread. While `TRACE` is present, the
    first step also opens one profiler session, and the finishing step closes it and logs the
    request's timeline."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        _state["on"] = not OFF.exists()
        if not _state["on"]:
            return fn(self, *args, **kwargs)
        _state.update(dict.fromkeys(_COUNTERS, 0))
        if _state["prof"] is None and TRACE.exists():
            _state["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            _state["prof"].start()
            _state["tracing"] = True
        _bounds["active"] = True
        _state["in_step"] = True
        start = time.perf_counter_ns()
        _mark("base")
        try:
            if _state["prof"] is None:
                result = fn(self, *args, **kwargs)
            else:
                with torch.profiler.record_function("isb_step"):
                    result = fn(self, *args, **kwargs)
        finally:
            _state["in_step"] = False
        _mark("between")
        end = time.perf_counter_ns()
        event = {"event": "step", "start": start, "end": end,
                 **{k: _state[k] for k in _COUNTERS}}
        outputs = result[0] if isinstance(result, tuple) else None
        event["finished"] = sum(1 for eco in (outputs or {}).values()
                                for out in eco.outputs if out.finished)
        if event["finished"]:
            _submit_bounds(end)
            if _state["prof"] is not None:
                torch.cuda.synchronize()
                _state["prof"].stop()
                _state["tracing"] = False
                event["trace"] = _step_trace(_state["prof"])
                _state["prof"] = None
        _log(event)
        return result
    return wrapper


def _arrival(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if _state["on"]:
            _log({"event": "arrival", "t": time.perf_counter_ns()})
        return fn(self, *args, **kwargs)
    return wrapper


def _wrap_hook_registration(method):
    """Register every forward hook / pre-hook inside an interp span."""
    @functools.wraps(method)
    def register(self, hook, *args, **kwargs):
        return method(self, _interp_span(hook), *args, **kwargs)
    return register


def install_engine():
    """Engine-process patches: steps, arrivals, interp and readout spans."""
    import torch.nn.functional as F
    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.v1.engine.core import EngineCore

    EngineCore.add_request = _arrival(EngineCore.add_request)
    EngineCore.step = _step(EngineCore.step)
    EngineCore.step_with_batch_queue = _step(EngineCore.step_with_batch_queue)
    Module = torch.nn.Module
    Module.register_forward_hook = _wrap_hook_registration(Module.register_forward_hook)
    Module.register_forward_pre_hook = _wrap_hook_registration(Module.register_forward_pre_hook)
    RMSNorm.forward = _readout_span(RMSNorm.__dict__.get("forward", CustomOp.forward))
    LogitsProcessor.forward = _readout_span(LogitsProcessor.forward)
    F.linear = _readout_span(F.linear)
    torch.Tensor.cpu = _readout_span(torch.Tensor.cpu)
    try:                                   # nnsight is installed only in the nnsight images
        from nnsight.intervention.interleaver import Interleaver
        from nnsight.modeling.vllm.interleaver import VLLMInterleaver
        from nnsight.modeling.vllm.model_runners.GPUModelRunner import Requests
    except ImportError:
        return
    Interleaver.handle = _interp_span(Interleaver.handle)
    # the taps path: its own handle, and the replay a breakable graph calls between its pieces
    VLLMInterleaver.handle = _interp_span(VLLMInterleaver.handle)
    VLLMInterleaver.replay = _interp_span(VLLMInterleaver.replay)
    for name in ("add", "refuse_chunked", "scope", "unflatten"):
        setattr(Requests, name, _interp_span(getattr(Requests, name)))
    for name in ("harvest", "record_saves"):      # packing saves for the return trip
        setattr(Requests, name, _interp_span(_readout_span(getattr(Requests, name))))


def _sent(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if _state["on"]:
            _client["sent"].append(time.perf_counter_ns())
        return fn(self, *args, **kwargs)
    return wrapper


def _sent_async(fn):
    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if _state["on"]:
            _client["sent"].append(time.perf_counter_ns())
        return await fn(self, *args, **kwargs)
    return wrapper


def _received(fn):
    @functools.wraps(fn)
    def wrapper(self, engine_core_outputs, *args, **kwargs):
        result = fn(self, engine_core_outputs, *args, **kwargs)
        if _state["on"] and any(out.finished for out in engine_core_outputs):
            _client["received"].append(time.perf_counter_ns())
        return result
    return wrapper


def install_client():
    from vllm.v1.engine.core_client import AsyncMPClient, InprocClient, SyncMPClient
    from vllm.v1.engine.output_processor import OutputProcessor

    InprocClient.add_request = _sent(InprocClient.add_request)   # engine in the client process
    SyncMPClient.add_request = _sent(SyncMPClient.add_request)
    AsyncMPClient.add_request_async = _sent_async(AsyncMPClient.add_request_async)
    OutputProcessor.process_outputs = _received(OutputProcessor.process_outputs)


def _events():
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line]


def _reset():
    LOG.unlink(missing_ok=True)
    _client["sent"].clear()
    _client["received"].clear()


def _trial(call):
    """One timed call: its six boundaries, the engine steps between them, and the request's
    boundary record (waited for after the call, outside the timed interval)."""
    from .timing import sync_cuda

    _reset()
    sync_cuda()
    t0 = time.perf_counter_ns()
    call()
    sync_cuda()
    t7 = time.perf_counter_ns()
    events = _events()
    arrivals = [e["t"] for e in events if e["event"] == "arrival"]
    steps = [e for e in events if e["event"] == "step"]
    finishing = [e for e in steps if e["finished"]]
    if not (_client["sent"] and _client["received"] and arrivals and finishing):
        raise RuntimeError(f"phase boundaries missing: {len(_client['sent'])} sent, "
                           f"{len(arrivals)} arrived, {len(finishing)} finishing steps, "
                           f"{len(_client['received'])} received")
    t1, t2 = _client["sent"][-1], arrivals[-1]
    t3, t4 = finishing[-1]["end"], _client["received"][-1]
    window = [e for e in steps if t2 <= e["start"] and e["end"] <= t3]
    deadline = time.monotonic() + 60
    while True:
        bounds = [e for e in events if e["event"] == "bounds" and e["end"] == t3]
        if bounds:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("the engine logged no boundary record for the finishing step")
        time.sleep(0.001)
        events = _events()
    return {"t": [t0, t1, t2, t3, t4, t7], "steps": window, "bounds": bounds[0],
            "requests": len(_client["sent"])}


def _stretches(t2, t3, bounds):
    """Engine time t2..t3 (ns) charged per segment ("engine", "hook", "readout"): each boundary's
    event time on the host clock, kept in order and inside the window, closes the stretch the
    previous region opened. Before the first boundary the engine is between steps."""
    marks, gpu_ms = bounds["marks"], bounds["gpu_ms"]
    end = t3 - t2                                   # times from here on are ns after t2
    offset = max(c - t2 - g * 1e6 for (c, _), g in zip(marks, gpu_ms))
    charged = dict.fromkeys(("engine", "hook", "readout"), 0.0)
    previous, kind = 0, "between"
    for (_, next_kind), g in zip(marks, gpu_ms):
        at = min(max(offset + g * 1e6, previous), end)
        charged[_SEGMENT[kind]] += at - previous
        previous, kind = at, next_kind
    charged[_SEGMENT[kind]] += end - previous
    return charged


def _summarize_traced(traced):
    """The traced call's GPU time per kind (ms), with the timeline it came from (logged by the
    step that finished the request); diagnostics and the span attribution's kernel time."""
    (timeline,) = [step["trace"] for step in traced["steps"] if "trace" in step]
    return {
        "kernels_ms": {kind: sum(b - a for a, b, k in timeline["kernels"] if k == kind) * 1e-3
                       for kind in _KINDS},
        "unmatched_kernel_ms": timeline["unmatched_kernel_us"] * 1e-3,
        "e2e_wall_ms": (traced["t"][-1] - traced["t"][0]) * 1e-6,
        "timeline": timeline,
    }


def _decompose(trial, traced):
    """One trial's subphases and raw segments under both attributions; `traced` is
    `_summarize_traced` of the row's traced call (span attribution only)."""
    t0, t1, t2, t3, t4, t7 = trial["t"]
    ms = 1e-6
    total = {k: sum(s[k] for s in trial["steps"]) * ms for k in _COUNTERS}
    charged = {k: v * ms for k, v in _stretches(t2, t3, trial["bounds"]).items()}
    kernels = traced["kernels_ms"]
    sub = {"client_prep": (t1 - t0) * ms, "request_transfer": (t2 - t1) * ms,
           "engine": (t3 - t2) * ms, "engine_outside_spans": charged["engine"],
           "engine_hook": charged["hook"], "engine_readout": charged["readout"],
           "engine_interp_host": total["interp_ns"], "engine_readout_host": total["readout_ns"],
           "output_transfer": (t4 - t3) * ms, "client_post": (t7 - t4) * ms}
    events = {
        "engine_outside_spans": charged["engine"],
        "hook_dispatch_and_edit": sub["client_prep"] + sub["request_transfer"] + charged["hook"],
        "readout_and_host_ipc": charged["readout"] + sub["output_transfer"] + sub["client_post"],
    }
    interp, readout = total["interp_ns"], total["readout_ns"]
    spans = {
        "base_gpu_kernels": kernels["base"],
        "engine_host": (sub["engine"] - kernels["base"] - interp - kernels["hook"]
                        - kernels["readout"]),
        "hook_dispatch_and_edit": (sub["client_prep"] + sub["request_transfer"]
                                   + interp - readout + kernels["hook"]),
        "readout_and_host_ipc": (readout + kernels["readout"] + sub["output_transfer"]
                                 + sub["client_post"]),
    }
    ordered = all(a <= b for a, b in zip(trial["t"], trial["t"][1:]))
    return {"e2e_wall_ms": (t7 - t0) * ms, "segments_raw_ms": events,
            "segments_raw_spans_ms": spans, "subphases_ms": sub, "boundaries_ordered": ordered,
            "requests": trial["requests"], "engine_steps": len(trial["steps"]),
            "bounds": trial["bounds"]}


def _warm(prepare, warmup):
    from .timing import force_gc

    for _ in range(warmup):
        prepare()()
        force_gc()


def _uninstrumented(prepare, *, warmup, n_trials):
    """Wall times (ms) of the same calls with every wrapper passing straight through, in the
    harness's timing loop (`isb.perf.timing.time_cell`): the baseline for the instrumentation's
    own cost. The engine process picks up `OFF` at its next step."""
    from .timing import force_gc, sync_cuda

    OFF.touch()
    _state["on"] = False
    try:
        _warm(prepare, warmup)
        times = []
        for _ in range(n_trials):
            call = prepare()
            sync_cuda()
            start = time.perf_counter_ns()
            call()
            sync_cuda()
            times.append((time.perf_counter_ns() - start) * 1e-6)
            force_gc()
    finally:
        OFF.unlink(missing_ok=True)
        _state["on"] = True
    return times


def profile_call(prepare, *, warmup, n_trials):
    """Time `n_trials` calls uninstrumented; then, instrumented: warm up, trace one call, warm up
    again (the first call after a profiler session is slow), and time `n_trials` calls; medians
    per field. Every call is prepared and followed by a collection outside its timed interval, as
    in `isb.perf.timing.time_cell`."""
    from .timing import force_gc

    plain = _uninstrumented(prepare, warmup=warmup, n_trials=n_trials)
    _warm(prepare, warmup)
    TRACE.touch()
    try:
        traced = _summarize_traced(_trial(prepare()))
    finally:
        TRACE.unlink(missing_ok=True)
    force_gc()
    _warm(prepare, warmup)
    trials = []
    for _ in range(n_trials):
        trials.append(_decompose(_trial(prepare()), traced))
        force_gc()

    def medians(field):
        return {k: statistics.median(t[field][k] for t in trials) for k in trials[0][field]}

    def gap(field):
        return max(abs(sum(t[field].values()) - t["e2e_wall_ms"]) for t in trials)

    e2e = statistics.median(t["e2e_wall_ms"] for t in trials)
    uninstrumented = statistics.median(plain)
    return {
        "e2e_wall_ms": e2e,
        "uninstrumented_e2e_ms": uninstrumented,
        "uninstrumented_trials_ms": plain,
        "instrumentation_overhead_fraction": e2e / uninstrumented - 1,
        "segments_raw_ms": medians("segments_raw_ms"),
        "segments_raw_spans_ms": medians("segments_raw_spans_ms"),
        "subphases_ms": medians("subphases_ms"),
        "traced": traced,
        "trials": trials,
        # Each trial's segments sum to its e2e by construction; the check is that every boundary
        # was captured in order, once per request, in every trial.
        "conservation": {
            "max_abs_gap_ms": gap("segments_raw_ms"),
            "max_abs_gap_spans_ms": gap("segments_raw_spans_ms"),
            "all_ordered": all(t["boundaries_ordered"] for t in trials),
            "single_request": all(t["requests"] == 1 for t in trials),
        },
    }


def main(job=Path("/job"), output=Path("/output")):
    import isb.methodologies  # noqa: F401  (registers the cells)
    from isb.jobs.contract import restore_spec
    from isb.jobs.entry import load
    from isb.sweep.execute import _bind_case, _task_params, _timed_units

    from .worker_memory import allow_callable_rpc

    spec, experiment = restore_spec(job)
    create_backend, _ = load(os.environ["ISB_BACKEND"])
    random.seed(experiment["seed"])                    # as isb.jobs.worker seeds before the backend
    torch.manual_seed(experiment["seed"])
    be, _ = create_backend(spec)
    allow_callable_rpc()
    model = be.load(spec.repo)
    rows = {}
    try:
        for regime in spec.regimes:
            if regime.kind not in ("interactive", "generation"):
                continue
            units = _timed_units(regime)
            new_tokens = regime.new_tokens or 1
            wanted = json.loads(os.environ.get("ISB_PHASE_ROWS") or "null")   # null: every row
            for label, params in [(task.label, task.params) for task in spec.tasks
                                  if wanted is None or task.label in wanted] + [
                    ("__vanilla__", None)]:
                key = f"{regime.name}|{label}"
                try:
                    if params is None:          # the backend's own request, nothing attached
                        prepare = functools.partial(functools.partial, be.plain, model, units,
                                                    new_tokens=new_tokens)
                    else:
                        call = _bind_case(spec, be.name, _task_params(regime, params), {})
                        prepare = functools.partial(call.prepare, be, model, units)
                    rows[key] = {"regime": regime.name, "label": label,
                                 "units_sha256": _units_sha(units), "new_tokens": new_tokens,
                                 **profile_call(prepare, warmup=spec.warmup,
                                                n_trials=spec.n_trials)}
                    print(f"[phase] {key}: e2e {rows[key]['e2e_wall_ms']:.2f} ms "
                          f"(uninstrumented {rows[key]['uninstrumented_e2e_ms']:.2f} ms) "
                          f"{json.dumps(rows[key]['segments_raw_ms'])}", flush=True)
                except Exception as error:      # isolated, as in the harness: later rows still run
                    traceback.print_exc()
                    rows[key] = {"error": repr(error)[:1000]}
    finally:
        be.teardown(model)
    output.mkdir(parents=True, exist_ok=True)
    import vllm

    (output / "phase_profile.json").write_text(json.dumps({
        "backend": os.environ["ISB_BACKEND"], "case": spec.name,
        "engine_options": json.loads(os.environ.get("ISB_ENGINE_OPTIONS") or "{}"),
        "vllm_version": vllm.__version__, "experiment_id": experiment["id"], "rows": rows},
        indent=1))


def _units_sha(units):
    import hashlib

    return hashlib.sha256(json.dumps(units, default=str).encode()).hexdigest()


STOCK = "vllm-plain/"          # the stock engine system: its configs are the floors


def _engine_ms(raw):
    """A trial's engine time outside interpretability code, under either attribution."""
    return sum(raw.values()) - raw["hook_dispatch_and_edit"] - raw["readout_and_host_ipc"]


def _floor_key(profile, row):
    """What a row and its floor share: the workload (the case name's first part), vLLM version,
    GPU count, timed units and token count."""
    return (profile["case"].split("/")[0], profile["vllm_version"], profile["gpus"],
            row["units_sha256"], row["new_tokens"])


ATTRIBUTIONS = {"events": "segments_raw_ms", "spans": "segments_raw_spans_ms"}


def split_with_floors(profiles, attribution="events"):
    """The figure's four segments for every profiled row, from the raw segments of `attribution`
    (a key of ATTRIBUTIONS) on both the row and its floor.

    `profiles` are loaded phase_profile.json files from one chain, each with the GPU count its
    config reserves (`gpus`, the tensor-parallel size of these engines). A row's floor is the plain
    request of a compiled `vllm-plain` profile of the same workload, vLLM version and GPU count,
    over the same timed units and token count; a row without one gets an error. Each trial is
    split, so every trial's segments still sum to its e2e; the record gives the per-segment
    medians and how far their sum falls from the median e2e."""
    field = ATTRIBUTIONS[attribution]
    floors = {}
    for profile in profiles:
        if profile["backend"].startswith(STOCK) and not profile["engine_options"].get(
                "enforce_eager"):
            for row in profile["rows"].values():
                if row.get("label") == "__vanilla__" and "error" not in row:
                    key = _floor_key(profile, row)
                    floors[key] = (profile["backend"], statistics.median(
                        _engine_ms(t[field]) for t in row["trials"]))
    out = []
    for profile in profiles:
        for key, row in profile["rows"].items():
            record = {"backend": profile["backend"], "case": profile["case"], "row": key,
                      "vllm_version": profile["vllm_version"], "gpus": profile["gpus"],
                      "attribution": attribution}
            out.append(record)
            if "error" in row:
                record["error"] = row["error"]
                continue
            match = floors.get(_floor_key(profile, row))
            if match is None:
                record["error"] = (f"no compiled {STOCK}* floor profiled for vLLM "
                                   f"{profile['vllm_version']} on {profile['gpus']} GPU(s) over "
                                   "these units")
                continue
            floor, floor_ms = match
            trials = []
            for trial in row["trials"]:
                raw = trial[field]
                trials.append({"e2e_wall_ms": trial["e2e_wall_ms"], "segments_ms": {
                    "base_compiled_engine": floor_ms,
                    "eager_graph_break_tax": _engine_ms(raw) - floor_ms,
                    "hook_dispatch_and_edit": raw["hook_dispatch_and_edit"],
                    "readout_and_host_ipc": raw["readout_and_host_ipc"]}})
            segments = {k: statistics.median(t["segments_ms"][k] for t in trials)
                        for k in trials[0]["segments_ms"]}
            gap = sum(segments.values()) - row["e2e_wall_ms"]
            record.update({
                "e2e_wall_ms": row["e2e_wall_ms"],
                "uninstrumented_e2e_ms": row["uninstrumented_e2e_ms"],
                "instrumentation_overhead_fraction": row["instrumentation_overhead_fraction"],
                "segments_ms": segments, "segments_raw_ms": row["segments_raw_ms"],
                "segments_raw_spans_ms": row["segments_raw_spans_ms"],
                "subphases_ms": row["subphases_ms"],
                "floor": floor, "floor_engine_ms": floor_ms, "trials": trials,
                "conservation": {
                    **row["conservation"],
                    "max_trial_gap_ms": max(abs(sum(t["segments_ms"].values()) - t["e2e_wall_ms"])
                                            for t in trials),
                    "median_sum_gap_ms": gap,
                    "median_sum_gap_fraction": gap / row["e2e_wall_ms"]}})
    return out


if os.environ.get(FLAG):
    # module level: runs in the client and again in a spawned engine process (`__mp_main__`)
    install_client()
    install_engine()

if __name__ == "__main__":
    main()

"""Phase profiler: where one cell call's wall time goes, measured inside one run (the latency
breakdown figure, paper docs/fig3-profiling-plan.md).

In-container entry: `python3 -m isb.perf.phase_profiler` with `ISB_PHASE_PROFILE=1`, the job
directory at /job and the output at /output (scripts/profile_phases.py launches it through the
backend's own compose.yml). It builds the backend exactly as `isb.jobs.entry` does, binds every
row of the job's spec the way the harness does (`isb.sweep.execute._bind_case`), and times each
row (or the labels listed in `ISB_PHASE_ROWS`, a JSON list) and the backend's plain request with
the harness's warm-up and trial counts. Each row runs three times in the same engine: with every
wrapper below passing straight through (no trace session open; the harness's timing loop), then
instrumented, then passing through again, so the record carries the instrumentation's own cost
against the mean of the two uninstrumented medians, and their gap shows how far the engine's own
timing moves over the row.

Host timestamps are `time.perf_counter_ns()` (CLOCK_MONOTONIC, one clock for every process in the
container), taken at the boundaries of each timed call (t1 to t4 once per request it sends):

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
the engine thread records one host time and the kind that starts there (a boundary; nothing else).

The GPU side comes from one torch.profiler session with CUDA activity only (CUPTI: every kernel,
copy and set with its GPU start and end, and the runtime or driver call that launched it, joined by
correlation id), open across a row's instrumented calls. Its clock is mapped onto perf_counter by
one constant offset: at the session's start and end the engine brackets ANCHORS stream queries with
perf_counter, each bracket bounds the offset, and the offset is the middle of all the bounds'
intersection (`_clock_fit`; its width is recorded per row). Each device activity takes the kind of the host region open when its
launching call started; an activity launched from another host thread, or without a launch record,
counts as base. Engine time t2..t3 is then charged instant by instant: while the GPU runs an
activity, to that activity's kind (the earlier-started one where two overlap); while the GPU is
idle, to the host region open at that instant, whose code the GPU is waiting for. Host work that
overlaps GPU work is charged to nothing, every instant is charged once, and the charges sum to
t3 - t2.

Subphases (they telescope to e2e = t7 - t0):
  client_prep  = t1 - t0     request_transfer = t2 - t1     engine = t3 - t2
  output_transfer = t4 - t3  client_post = t7 - t4
  engine_hook, engine_readout = the engine time charged to hook / readout
  engine_outside_spans = engine - engine_hook - engine_readout
A call that sends several requests, one after another, sums each subphase over its requests,
with client_prep also taking the client time from each request's t4 to the next one's t1; each
request's engine window t2..t3 is charged on its own.

Raw per-run segments, kernel-trace attribution (`segments_raw_ms`):
  engine_outside_spans   = the base model and the engine's own host work, together
  hook_dispatch_and_edit = client_prep + request_transfer + engine_hook
  readout_and_host_ipc   = engine_readout + output_transfer + client_post

The span attribution (`segments_raw_spans_ms`), kept for comparison, counts host span time on top
of the GPU time it overlaps:
  base_gpu_kernels       = base kernel time
  engine_host            = engine - base - interp - hook_kernels - readout_kernels
  hook_dispatch_and_edit = client_prep + request_transfer + (interp - readout) + hook_kernels
  readout_and_host_ipc   = readout + readout_kernels + output_transfer + client_post
(interp and readout here are the host time inside those regions; kernel times are summed per kind.)

`split_with_floors` turns raw segments into the figure's four with the compiled stock floor of
the same workload, vLLM version and GPU count (a `vllm-plain` config without enforce_eager,
profiled in the same chain over the same units). The floor's plain request gives F, its engine
time outside interpretability code (engine_outside_spans; the same attribution on both sides):
  base_compiled_engine  = F x the requests the call sent (`forwards_per_call`)
  eager_graph_break_tax = engine time outside interpretability code - base_compiled_engine
      (signed; it carries both a forced-eager engine and CUDA graphs run without torch.compile)
  hook_dispatch_and_edit, readout_and_host_ipc: as measured

Patches go in when this module is imported with ISB_PHASE_PROFILE set: in the client before the
engine starts, and in the engine process because vLLM starts it with fork (inherited) or spawn
(which re-imports this main module). Without an open session every wrapper passes straight
through after one flag check; under torch.compile tracing they pass through too, so compiled
graphs are unchanged; no boundary is recorded while a stream is capturing a CUDA graph.
`ISB_PHASE_MARKS=0` keeps the session but records no boundaries: the tracing's own cost.
"""
from __future__ import annotations

import bisect
import functools
import json
import os
import random
import statistics
import threading
import time
import traceback
from pathlib import Path

import torch

FLAG = "ISB_PHASE_PROFILE"
LOG = Path("/dev/shm/isb_phase_profile.jsonl")      # engine process -> client: arrivals, steps
TRACE = Path("/dev/shm/isb_phase_trace")            # present: the engine opens a trace session
STOP = Path("/dev/shm/isb_phase_stop")              # present: the engine closes it and exports
EXPORT = Path("/dev/shm/isb_phase_export.json")     # the closed session's boundaries and activities
MARKS = os.environ.get("ISB_PHASE_MARKS", "1") != "0"
_KINDS = ("base", "hook", "readout")
_SEGMENT = {"between": "engine", "base": "engine", "hook": "hook", "readout": "readout"}
_perf = time.perf_counter_ns
_compiling = torch.compiler.is_compiling

# engine process: the open session, and the boundaries of the request in flight
_session = None
_live = False            # boundaries are recorded: a session is open and no graph is capturing
_in_step = False
_depth = 0               # an interpretability span is open (only the outermost one marks)
_reading = 0             # a readout span is open
_owner = None            # the thread running engine steps; spans on other threads pass through
_marks = []
_append = _marks.append
_finished = []           # requests finished while the session is open: {"end", "marks"}
_anchors = []            # per session end: {anchor call: [(before, after)] perf_counter}
# client process
_on = True               # the client wrappers record (off for the uninstrumented pass)
_client = {"sent": [], "received": []}


def _capturing():
    return torch.cuda.is_current_stream_capturing()


def _interp_span(fn):
    """Run `fn` as interpretability-system code: a boundary at its start (hook) and at its end
    (back to base, or between outside a step). Nested spans mark once."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _depth
        if not _live or _depth or threading.get_ident() != _owner or _compiling():
            return fn(*args, **kwargs)
        _depth = 1
        _append((_perf(), "hook"))
        try:
            return fn(*args, **kwargs)
        finally:
            _append((_perf(), "base" if _in_step else "between"))
            _depth = 0
    return wrapper


def _readout_span(fn):
    """Run `fn` as readout when it runs inside an interp span; otherwise (the base model's own
    calls, and everything while no session is open) pass straight through after one check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _reading
        if not _depth or _reading or threading.get_ident() != _owner or _compiling():
            return fn(*args, **kwargs)
        _reading = 1
        _append((_perf(), "readout"))
        try:
            return fn(*args, **kwargs)
        finally:
            _append((_perf(), "hook"))
            _reading = 0
    return wrapper


def _log(event):
    """Append one line with a single write() on an O_APPEND descriptor, so a line from another
    thread or process never lands inside it."""
    line = (json.dumps(event) + "\n").encode()
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, line)
    finally:
        os.close(fd)
    if written != len(line):
        raise RuntimeError(f"short write to {LOG}: {written} of {len(line)} bytes")


ANCHORS = 20                   # bracketed calls at each end of a session (clock alignment)
_ANCHOR_CALLS = ("cudaStreamQuery", "cudaDeviceSynchronize")


def _anchor_round():
    """ANCHORS non-blocking stream queries, then ANCHORS synchronizes, each bracketed by
    perf_counter: the trace records every call, and each bracket bounds the clock offset. The
    synchronizes are the fallback should the trace leave the queries out."""
    query = torch.cuda.current_stream().query
    rounds = {}
    for name, call in zip(_ANCHOR_CALLS, (query, torch.cuda.synchronize)):
        brackets = []
        for _ in range(ANCHORS):
            before = _perf()
            call()
            brackets.append((before, _perf()))
        rounds[name] = brackets
    return rounds


def _open_session():
    global _session
    _session = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
    _session.start()
    _anchors[:] = [_anchor_round()]
    _finished.clear()


def _close_session():
    """Close the session and export, for the client, every finished request's boundaries and
    every device activity on the perf_counter clock."""
    global _session, _live, _marks, _append
    _anchors.append(_anchor_round())
    prof, _session = _session, None
    _live = False
    _marks = []
    _append = _marks.append
    prof.stop()
    export = json.dumps(_export(prof.events(), _anchors, _finished))
    _finished.clear()
    partial = EXPORT.with_suffix(".partial")
    partial.write_text(export)
    partial.rename(EXPORT)


def _clock_fit(host, anchors):
    """The constant offset perf_counter minus trace clock (ns). A bracketed call's record lies
    inside its bracket, so each pair bounds the offset to [before - record start, after - record
    end]; the session's first ANCHORS records of the call pair with the opening round and its last
    ANCHORS with the closing one, and the offset is the middle of all the bounds' intersection.
    Its width is the fit's accuracy (negative: the bounds disagree by that much)."""
    for name in _ANCHOR_CALLS:
        records = [e for e in host if e.name == name]
        if len(records) >= 2 * ANCHORS:
            break
    else:
        raise RuntimeError("the trace holds fewer than "
                           f"{2 * ANCHORS} records of every anchor call {_ANCHOR_CALLS}")

    def bounds(brackets, matched):
        return (max(b - r.time_range.start * 1e3 for (b, _), r in zip(brackets, matched)),
                min(a - r.time_range.end * 1e3 for (_, a), r in zip(brackets, matched)))

    (lo1, hi1) = bounds(anchors[0][name], records[:ANCHORS])
    (lo2, hi2) = bounds(anchors[-1][name], records[-ANCHORS:])
    lo, hi = max(lo1, lo2), min(hi1, hi2)
    fit = {"call": name, "bracket_width_ns": hi - lo, "start_width_ns": hi1 - lo1,
           "end_width_ns": hi2 - lo2, "end_minus_start_ns": (lo2 + hi2 - lo1 - hi1) / 2,
           "narrowest_bracket_ns": min(a - b for r in (anchors[0], anchors[-1])
                                       for b, a in r[name]),
           "span_s": (records[-1].time_range.start - records[0].time_range.start) * 1e-6}
    return (lo + hi) / 2, {records[0].thread, records[-1].thread}, fit


def _export(events, anchors, finished):
    """The session's device activities as [gpu start, gpu end, launch or None, launched by the
    engine thread] in perf_counter ns, with the clock fit and trace diagnostics."""
    from torch.autograd import DeviceType

    host = sorted((e for e in events if e.device_type == DeviceType.CPU),
                  key=lambda e: e.time_range.start)
    offset, engine_threads, fit = _clock_fit(host, anchors)

    def perf(us):                    # whole ns: perf_counter values are beyond float precision
        return round(us * 1e3 + offset)

    launches = {e.id: e for e in host if e.name.startswith("cu")}
    activities = []
    for e in events:
        if e.device_type != DeviceType.CUDA or e.is_user_annotation:
            continue
        launch = launches.get(e.id)
        activities.append([perf(e.time_range.start), perf(e.time_range.end),
                           None if launch is None else perf(launch.time_range.start),
                           launch is not None and launch.thread in engine_threads])
    return {"requests": finished, "activities": activities, "alignment": fit,
            "trace": {"launch_records": len(launches), "device_activities": len(activities),
                      "other_host_records": sum(1 for e in host if not e.name.startswith("cu")),
                      "launches_after_gpu_start": sum(
                          1 for a in activities if a[2] is not None and a[2] > a[0])}}


def _step(fn):
    """An engine step. Without a session it passes straight through after one file check; the
    first step that finds `TRACE` opens the session, and the first step that finds `STOP` closes
    it (before running). In a session, the step's start and end are boundaries, and the step that
    finishes a request files that request's boundaries under its end time."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        global _live, _in_step, _owner, _marks, _append
        if _session is None:
            if not TRACE.exists() or STOP.exists():
                return fn(self, *args, **kwargs)
            _open_session()
        elif STOP.exists():
            _close_session()
            return fn(self, *args, **kwargs)
        _owner = threading.get_ident()
        _live = MARKS and not _capturing()
        _in_step = True
        start = _perf()
        if _live:
            _append((start, "base"))
        try:
            result = fn(self, *args, **kwargs)
        finally:
            _in_step = False
        end = _perf()
        if _live:
            _append((end, "between"))
        outputs = result[0] if isinstance(result, tuple) else None
        finished = sum(1 for eco in (outputs or {}).values() for out in eco.outputs
                       if out.finished)
        if finished:
            _finished.append({"end": end, "marks": _marks})
            _marks = []
            _append = _marks.append
            _live = False
        _log({"event": "step", "start": start, "end": end, "finished": finished})
        return result
    return wrapper


def _arrival(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if _session is not None:
            _log({"event": "arrival", "t": _perf()})
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
        if _on:
            _client["sent"].append(_perf())
        return fn(self, *args, **kwargs)
    return wrapper


def _sent_async(fn):
    @functools.wraps(fn)
    async def wrapper(self, *args, **kwargs):
        if _on:
            _client["sent"].append(_perf())
        return await fn(self, *args, **kwargs)
    return wrapper


def _received(fn):
    @functools.wraps(fn)
    def wrapper(self, engine_core_outputs, *args, **kwargs):
        result = fn(self, engine_core_outputs, *args, **kwargs)
        if _on and any(out.finished for out in engine_core_outputs):
            _client["received"].append(_perf())
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
    """Every complete line of the log. A line the engine is still writing has no newline yet; it
    is left for the next read."""
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().split("\n")[:-1] if line]


def _reset():
    LOG.unlink(missing_ok=True)
    _client["sent"].clear()
    _client["received"].clear()


def _trial(call):
    """One timed call: its boundaries and the engine steps of its requests. The requests'
    region boundaries and the GPU activities arrive with the session's export."""
    from .timing import sync_cuda

    _reset()
    sync_cuda()
    t0 = _perf()
    call()
    sync_cuda()
    t7 = _perf()
    return _bounds(t0, t7, _client["sent"], _client["received"], _events())


def _bounds(t0, t7, sent, received, events):
    """A call's boundaries: t0, t7 and, per request in the order sent, [t1, t2, t3, t4]. The
    requests run one after another, so the k-th send, arrival, finishing step and finished output
    belong to the k-th request; counts that do not pair up raise."""
    arrivals = [e["t"] for e in events if e["event"] == "arrival"]
    steps = [e for e in events if e["event"] == "step"]
    finishing = [e["end"] for e in steps if e["finished"]]
    if not (sent and len(sent) == len(arrivals) == len(finishing) == len(received)):
        raise RuntimeError(f"phase boundaries do not pair up: {len(sent)} sent, "
                           f"{len(arrivals)} arrived, {len(finishing)} finishing steps, "
                           f"{len(received)} received")
    requests = [list(r) for r in zip(sent, arrivals, finishing, received)]
    window = [e for e in steps if any(t2 <= e["start"] and e["end"] <= t3
                                      for _, t2, t3, _ in requests)]
    return {"t0": t0, "t7": t7, "requests": requests, "steps": window}


def _attribute(t2, t3, marks, activities):
    """Engine time t2..t3 (ns) charged per segment ("engine", "hook", "readout"), with the device
    activities inside the window and their kinds. `marks`: (host time, kind starting there);
    `activities`: [gpu start, gpu end, launch or None, launched by the engine thread]."""
    times = [c for c, _ in marks]
    kinds = [k for _, k in marks]
    charged = dict.fromkeys(("engine", "hook", "readout"), 0.0)

    def region(t):
        i = bisect.bisect_right(times, t) - 1
        return kinds[i] if i >= 0 else "between"

    def idle(a, b):                     # the GPU waits: charge the host regions open meanwhile
        i = bisect.bisect_right(times, a) - 1
        while a < b:
            nxt = times[i + 1] if i + 1 < len(times) else b
            stop = min(b, nxt)
            charged[_SEGMENT[kinds[i] if i >= 0 else "between"]] += stop - a
            a, i = stop, i + 1

    placed, unmatched, other = [], 0.0, 0.0
    for start, end, launch, engine_thread in activities:
        a, b = max(start, t2), min(end, t3)
        if a >= b:
            continue
        if launch is None:
            kind, unmatched = "base", unmatched + b - a
        elif not engine_thread:
            kind, other = "base", other + b - a
        else:
            kind = region(launch)
        placed.append((a, b, kind))
    placed.sort()
    covered = t2
    for a, b, kind in placed:
        if b <= covered:
            continue
        if a > covered:
            idle(covered, a)
        charged[_SEGMENT[kind]] += b - max(a, covered)
        covered = b
    idle(covered, t3)
    return charged, placed, unmatched, other


def _host_time(t2, t3, marks):
    """Host time (ns) inside hook and readout regions within t2..t3."""
    out = {"hook": 0.0, "readout": 0.0}
    for (c, kind), (n, _) in zip(marks, marks[1:] + [(t3, None)]):
        if kind in out:
            out[kind] += max(0, min(n, t3) - max(c, t2))
    return out


def _decompose(trial, marks, activities):
    """One trial's subphases and raw segments under both attributions. `marks` maps a finishing
    step's end to the boundaries of the request it finished; each request's engine window
    t2..t3 is charged on its own and the charges add up."""
    t0, t7, requests = trial["t0"], trial["t7"], trial["requests"]
    ms = 1e-6
    charged = dict.fromkeys(("engine", "hook", "readout"), 0.0)
    host = {"hook": 0.0, "readout": 0.0}
    placed, unmatched, other, all_marks = [], 0.0, 0.0, []
    for _, t2, t3, _ in requests:
        own = marks[t3]
        c, p, u, o = _attribute(t2, t3, own, activities)
        for k, v in c.items():
            charged[k] += v
        for k, v in _host_time(t2, t3, own).items():
            host[k] += v
        placed, unmatched, other, all_marks = placed + p, unmatched + u, other + o, all_marks + own
    charged = {k: v * ms for k, v in charged.items()}
    host = {k: v * ms for k, v in host.items()}
    kernels = dict.fromkeys(_KINDS, 0.0)
    for a, b, kind in placed:
        kernels["base" if kind == "between" else kind] += (b - a) * ms
    interp, readout = host["hook"] + host["readout"], host["readout"]
    # client time before each send (from t0 or the previous request's t4) prepares that request;
    # client time after the last request's t4 handles the result
    previous = [t0] + [t4 for *_, t4 in requests[:-1]]
    sub = {"client_prep": sum(t1 - p for p, (t1, *_) in zip(previous, requests)) * ms,
           "request_transfer": sum(t2 - t1 for t1, t2, _, _ in requests) * ms,
           "engine": sum(t3 - t2 for _, t2, t3, _ in requests) * ms,
           "engine_outside_spans": charged["engine"],
           "engine_hook": charged["hook"], "engine_readout": charged["readout"],
           "engine_interp_host": interp, "engine_readout_host": readout,
           "output_transfer": sum(t4 - t3 for _, _, t3, t4 in requests) * ms,
           "client_post": (t7 - requests[-1][3]) * ms}
    events = {
        "engine_outside_spans": charged["engine"],
        "hook_dispatch_and_edit": sub["client_prep"] + sub["request_transfer"] + charged["hook"],
        "readout_and_host_ipc": charged["readout"] + sub["output_transfer"] + sub["client_post"],
    }
    spans = {
        "base_gpu_kernels": kernels["base"],
        "engine_host": (sub["engine"] - kernels["base"] - interp - kernels["hook"]
                        - kernels["readout"]),
        "hook_dispatch_and_edit": (sub["client_prep"] + sub["request_transfer"]
                                   + interp - readout + kernels["hook"]),
        "readout_and_host_ipc": (readout + kernels["readout"] + sub["output_transfer"]
                                 + sub["client_post"]),
    }
    flat = [t0] + [t for request in requests for t in request] + [t7]
    ordered = all(a <= b for a, b in zip(flat, flat[1:]))
    t2 = requests[0][1]                                 # the timeline's origin
    return {"e2e_wall_ms": (t7 - t0) * ms, "segments_raw_ms": events,
            "segments_raw_spans_ms": spans, "subphases_ms": sub, "boundaries_ordered": ordered,
            "requests": len(requests), "engine_steps": len(trial["steps"]),
            "marks": len(all_marks), "kernels_ms": kernels, "unmatched_kernel_ms": unmatched * ms,
            "other_thread_kernel_ms": other * ms,
            "timeline": {"marks": [[(c - t2) * 1e-3, k] for c, k in all_marks],
                         "activities": [[(a - t2) * 1e-3, (b - t2) * 1e-3, k]
                                        for a, b, k in placed]}}


def _warm(prepare, warmup):
    from .timing import force_gc

    for _ in range(warmup):
        prepare()()
        force_gc()


def _uninstrumented(prepare, *, warmup, n_trials):
    """Wall times (ms) of the same calls with no session open, so every engine wrapper passes
    straight through, in the harness's timing loop (`isb.perf.timing.time_cell`): the baseline for
    the instrumentation's own cost."""
    global _on
    from .timing import force_gc, sync_cuda

    _on = False
    try:
        _warm(prepare, warmup)
        times = []
        for _ in range(n_trials):
            call = prepare()
            sync_cuda()
            start = _perf()
            call()
            sync_cuda()
            times.append((_perf() - start) * 1e-6)
            force_gc()
    finally:
        _on = True
    return times


def _close(prepare, timeout=600):
    """Have the engine close its session: one more untimed call, whose first step exports."""
    TRACE.unlink(missing_ok=True)
    STOP.touch()
    try:
        prepare()()
        deadline = time.monotonic() + timeout
        while not EXPORT.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("the engine exported no trace session")
            time.sleep(0.01)
        export = json.loads(EXPORT.read_text())
    finally:
        STOP.unlink(missing_ok=True)
        EXPORT.unlink(missing_ok=True)
    return export


def profile_call(prepare, *, warmup, n_trials):
    """Time `n_trials` calls uninstrumented; then open a trace session, warm up (the session
    opens in the first of these calls), time `n_trials` calls instrumented and close the session;
    then time `n_trials` calls uninstrumented again. Every trial is decomposed from the export.
    Every call is prepared and followed by a collection outside its timed interval, as in
    `isb.perf.timing.time_cell`."""
    from .timing import force_gc

    plain = _uninstrumented(prepare, warmup=warmup, n_trials=n_trials)
    TRACE.touch()
    try:
        _warm(prepare, warmup + 1)
        raw = []
        for _ in range(n_trials):
            raw.append(_trial(prepare()))
            force_gc()
        export = _close(prepare)
    finally:
        TRACE.unlink(missing_ok=True)
    plain_after = _uninstrumented(prepare, warmup=warmup, n_trials=n_trials)
    marks = {r["end"]: r["marks"] for r in export["requests"]}
    trials = [_decompose(t, marks, export["activities"]) for t in raw]
    # the trial whose e2e is the median keeps its timeline, for diagnostics
    chosen = sorted(range(n_trials), key=lambda i: trials[i]["e2e_wall_ms"])[(n_trials - 1) // 2]
    timeline = {"trial": chosen, **trials[chosen]["timeline"]}
    for t in trials:
        del t["timeline"]

    def medians(field):
        return {k: statistics.median(t[field][k] for t in trials) for k in trials[0][field]}

    def gap(field):
        return max(abs(sum(t[field].values()) - t["e2e_wall_ms"]) for t in trials)

    e2e = statistics.median(t["e2e_wall_ms"] for t in trials)
    before, after = statistics.median(plain), statistics.median(plain_after)
    uninstrumented = (before + after) / 2
    return {
        "e2e_wall_ms": e2e,
        "uninstrumented_e2e_ms": uninstrumented,
        "uninstrumented_before_e2e_ms": before,
        "uninstrumented_after_e2e_ms": after,
        "uninstrumented_gap_fraction": after / before - 1,
        "uninstrumented_trials_ms": plain,
        "uninstrumented_after_trials_ms": plain_after,
        "instrumentation_overhead_fraction": e2e / uninstrumented - 1,
        "segments_raw_ms": medians("segments_raw_ms"),
        "segments_raw_spans_ms": medians("segments_raw_spans_ms"),
        "subphases_ms": medians("subphases_ms"),
        "kernels_ms": medians("kernels_ms"),
        "alignment": export["alignment"],
        "trace": export["trace"],
        "timeline": timeline,
        "trials": trials,
        # Each trial's segments sum to its e2e by construction; the check is that every boundary
        # was captured in order, once per request, in every trial. A call may send several
        # requests one after another (`requests` per trial); single_request: every call sent one.
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
    split, so every trial's segments still sum to its e2e; the record reports the trial whose
    e2e is the median, with every trial kept. A trial's base is F for each request its call sent
    (a call that reads a value in one request and writes it in the next runs the engine twice), and
    the record's `forwards_per_call` is the reported trial's request count."""
    field = ATTRIBUTIONS[attribution]
    floors = {}
    for profile in profiles:
        if profile["backend"].startswith(STOCK) and not profile["engine_options"].get(
                "enforce_eager"):
            for row in profile["rows"].values():
                if row.get("label") == "__vanilla__" and "error" not in row:
                    key = _floor_key(profile, row)
                    floors[key] = (profile["backend"], statistics.median(
                        _engine_ms(t[field]) / t["requests"] for t in row["trials"]))
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
                raw, base = trial[field], floor_ms * trial["requests"]
                trials.append({"e2e_wall_ms": trial["e2e_wall_ms"], "segments_ms": {
                    "base_compiled_engine": base,
                    "eager_graph_break_tax": _engine_ms(raw) - base,
                    "hook_dispatch_and_edit": raw["hook_dispatch_and_edit"],
                    "readout_and_host_ipc": raw["readout_and_host_ipc"]}})
            # the trial whose e2e is the median (the lower middle one for an even count): its
            # segments sum to its own e2e
            ordered = sorted(range(len(trials)), key=lambda i: trials[i]["e2e_wall_ms"])
            chosen = ordered[(len(ordered) - 1) // 2]
            segments = trials[chosen]["segments_ms"]
            record.update({
                "e2e_wall_ms": trials[chosen]["e2e_wall_ms"], "reported_trial": chosen,
                "marks": row["trials"][chosen]["marks"],
                "forwards_per_call": row["trials"][chosen]["requests"],
                "uninstrumented_e2e_ms": row["uninstrumented_e2e_ms"],
                "uninstrumented_before_e2e_ms": row["uninstrumented_before_e2e_ms"],
                "uninstrumented_after_e2e_ms": row["uninstrumented_after_e2e_ms"],
                "uninstrumented_gap_fraction": row["uninstrumented_gap_fraction"],
                "instrumentation_overhead_fraction": row["instrumentation_overhead_fraction"],
                "clock_bracket_width_ns": row["alignment"]["bracket_width_ns"],
                "segments_ms": segments, "segments_raw_ms": row["segments_raw_ms"],
                "segments_raw_spans_ms": row["segments_raw_spans_ms"],
                "subphases_ms": row["subphases_ms"],
                "floor": floor, "floor_engine_ms": floor_ms, "trials": trials,
                "conservation": {
                    **row["conservation"],
                    "max_trial_gap_ms": max(abs(sum(t["segments_ms"].values()) - t["e2e_wall_ms"])
                                            for t in trials)}})
    return out


if os.environ.get(FLAG):
    # module level: runs in the client and again in a spawned engine process (`__mp_main__`)
    install_client()
    install_engine()

if __name__ == "__main__":
    main()

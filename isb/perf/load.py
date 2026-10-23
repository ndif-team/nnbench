"""Load-pattern timing (design.md §12.17): how standard inference benchmarks issue requests.

`run_offline` is MLPerf Offline / `vllm bench throughput`: the units go out as one sweep, one call
per chunk, chunks back to back. `run_concurrent` is the closed-loop serving pattern (`vllm bench
serve --max-concurrency`, AIPerf's concurrency sweep): `concurrency` clients each issue one unit
per call and the next as soon as theirs returns, until the units run out. A backend whose interface
takes one request at a time (an HF forward, vLLM's offline `LLM` class) has its requests queued in
front of it, one at a time, as a server built on that interface would; the time a request waits in
that queue counts toward its latency, as a client would see it. `run_open` is the open-loop
pattern (MLPerf Server, `vllm bench serve --request-rate`): each unit arrives at a fixed time of a
seeded Poisson process and is issued then on its own thread, whatever the system is doing, so
above capacity the requests queue and their latency grows.

All run one untimed warm-up pass over every unit, under the same pattern as the timed passes (so
the shapes the engine sees with requests overlapping are warm), then `passes` timed passes. A pass
is timed between CUDA syncs; a request from issue to return, or open loop from its scheduled
arrival, so a late issue counts against the system. Call preparation (the fresh parameter copy)
happens outside both. The outputs of the first `keep` units of the last pass are returned for
scoring.
"""
from __future__ import annotations

import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .timing import force_gc, peak_mem_mb, reset_peak_mem, sync_cuda


@dataclass
class LoadResult:
    units: int                       # units per pass
    calls: int                       # calls per pass
    concurrency: int                 # clients (1 offline, 0 open loop)
    serial: bool                     # requests were queued one at a time in front of the interface
    pass_ms: list = field(default_factory=list)
    request_ms: list = field(default_factory=list)
    peak_mem_mb: float = 0.0
    offered_rate: float | None = None   # open loop: arrivals per second
    max_in_flight: int | None = None    # open loop: most requests issued and not yet returned

    @property
    def median_pass_ms(self) -> float:
        return statistics.median(self.pass_ms)

    @property
    def throughput(self) -> float:
        """Units per second over the median pass."""
        return self.units / (self.median_pass_ms / 1000.0)

    def percentile(self, q: float) -> float:
        ordered = sorted(self.request_ms)
        rank = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
        return ordered[rank]

    def summary(self) -> dict:
        return {"units": self.units, "calls_per_pass": self.calls, "concurrency": self.concurrency,
                "serial": self.serial, "passes": len(self.pass_ms),
                "pass_ms": list(self.pass_ms), "median_pass_ms": self.median_pass_ms,
                "request_p50_ms": self.percentile(50), "request_p90_ms": self.percentile(90),
                "request_p99_ms": self.percentile(99),
                "request_mean_ms": statistics.fmean(self.request_ms),
                **({"offered_rate": self.offered_rate, "max_in_flight": self.max_in_flight}
                   if self.offered_rate is not None else {})}


def run_offline(prepare: Callable[[list], Callable[[], Any]], chunks: list, *, passes: int,
                keep: int):
    """Issue `chunks` (lists of units) back to back, one call each. Returns (LoadResult, outputs),
    `outputs` mapping chunk index -> output for the chunks that cover the first `keep` units."""
    if passes < 1 or not chunks:
        raise ValueError("offline needs passes>=1 and at least one chunk")
    result = LoadResult(units=sum(len(c) for c in chunks), calls=len(chunks), concurrency=1,
                        serial=True)
    covering, seen = set(), 0
    for index, chunk in enumerate(chunks):
        if seen < keep:
            covering.add(index)
        seen += len(chunk)
    outputs = {}
    for timed in [False] + [True] * passes:
        if timed:
            force_gc()
            reset_peak_mem()
        calls = [prepare(chunk) for chunk in chunks]
        sync_cuda()
        start = perf_counter()
        for index, call in enumerate(calls):
            t0 = perf_counter()
            out = call()
            if timed:
                result.request_ms.append((perf_counter() - t0) * 1000.0)
            if index in covering:
                outputs[index] = out
        sync_cuda()
        if timed:
            result.pass_ms.append((perf_counter() - start) * 1000.0)
    result.peak_mem_mb = peak_mem_mb()
    return result, outputs


def run_concurrent(prepare: Callable[[Any], Callable[[], Any]], units: list, *, concurrency: int,
                   passes: int, keep: int, serial: bool):
    """Closed loop: `concurrency` client threads share the unit list, each issuing its next unit
    as soon as its previous one returns. Returns (LoadResult, outputs), `outputs` mapping unit
    index -> output for the first `keep` units of the last pass. One untimed warm-up pass over
    every unit, then `passes` timed passes. The first error a client meets is raised once every
    client has stopped."""
    if passes < 1 or concurrency < 1 or not units:
        raise ValueError("concurrent needs passes>=1, concurrency>=1 and at least one unit")
    result = LoadResult(units=len(units), calls=len(units), concurrency=concurrency, serial=serial)
    gate = threading.Lock() if serial else None
    outputs = {}

    def sweep(indices, timed):
        calls = {index: prepare(units[index]) for index in indices}   # outside the pass timer
        pending = iter(indices)
        take = threading.Lock()
        errors, latencies, produced = [], [], {}

        def client():
            while not errors:
                with take:
                    index = next(pending, None)
                if index is None:
                    return
                try:
                    t0 = perf_counter()
                    if gate is None:
                        out = calls[index]()
                    else:
                        with gate:
                            out = calls[index]()
                    latencies.append((perf_counter() - t0) * 1000.0)
                    if index < keep:
                        produced[index] = out
                except BaseException as error:          # stop the sweep; re-raised below
                    errors.append(error)
                    return

        threads = [threading.Thread(target=client, name=f"isb-client-{i}", daemon=True)
                   for i in range(min(concurrency, len(indices)))]
        sync_cuda()
        start = perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        sync_cuda()
        elapsed = (perf_counter() - start) * 1000.0
        if errors:
            raise errors[0]
        if timed:
            result.pass_ms.append(elapsed)
            result.request_ms.extend(latencies)
        return produced

    sweep(list(range(len(units))), timed=False)                 # warm-up pass, untimed
    force_gc()
    reset_peak_mem()
    for _ in range(passes):
        outputs = sweep(list(range(len(units))), timed=True)
    result.peak_mem_mb = peak_mem_mb()
    return result, outputs


def arrival_offsets(count: int, rate: float, seed: int = 0) -> list:
    """Seconds from the start of a pass at which each of `count` requests arrives: a Poisson
    process at `rate` per second (exponential gaps), the first at 0, the same for every pass."""
    rng = random.Random(seed)
    offsets, t = [], 0.0
    for _ in range(count):
        offsets.append(t)
        t += rng.expovariate(rate)
    return offsets


def run_open(prepare: Callable[[Any], Callable[[], Any]], units: list, *, rate: float, passes: int,
             keep: int, serial: bool):
    """Open loop: unit i is issued at `arrival_offsets(...)[i]` on a thread of its own, whether or
    not earlier requests have returned. Returns (LoadResult, outputs), `outputs` mapping unit index
    -> output for the first `keep` units of the last pass. One untimed warm-up pass over every unit
    at the same arrivals, then `passes` timed passes. The first error a request meets stops
    further arrivals and is raised once the issued requests have returned."""
    if passes < 1 or rate <= 0 or not units:
        raise ValueError("open loop needs passes>=1, rate>0 and at least one unit")
    result = LoadResult(units=len(units), calls=len(units), concurrency=0, serial=serial,
                        offered_rate=rate, max_in_flight=0)
    gate = threading.Lock() if serial else None
    offsets = arrival_offsets(len(units), rate)

    def sweep():
        calls = [prepare(unit) for unit in units]               # outside the pass timer
        count = threading.Lock()
        errors, latencies, produced, state = [], [], {}, {"in_flight": 0, "peak": 0}

        def request(index, due):
            try:
                if gate is None:
                    out = calls[index]()
                else:
                    with gate:
                        out = calls[index]()
                latencies.append((perf_counter() - due) * 1000.0)
                if index < keep:
                    produced[index] = out
            except BaseException as error:          # stop further arrivals; re-raised below
                errors.append(error)
            finally:
                with count:
                    state["in_flight"] -= 1

        # ponytail: one thread per request, so a pass holds at most len(units) threads; a worker
        # pool would be needed only for sweeps far larger than the suites' 128 units
        threads = []
        sync_cuda()
        start = perf_counter()
        for index, offset in enumerate(offsets):
            if errors:
                break
            due = start + offset
            delay = due - perf_counter()
            if delay > 0:
                time.sleep(delay)
            with count:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            thread = threading.Thread(target=request, args=(index, due),
                                      name=f"isb-arrival-{index}", daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        sync_cuda()
        elapsed = (perf_counter() - start) * 1000.0
        if errors:
            raise errors[0]
        return elapsed, latencies, produced, state["peak"]

    sweep()                                                     # warm-up pass, untimed
    force_gc()
    reset_peak_mem()
    outputs = {}
    for _ in range(passes):
        elapsed, latencies, outputs, peak = sweep()
        result.pass_ms.append(elapsed)
        result.request_ms.extend(latencies)
        result.max_in_flight = max(result.max_in_flight, peak)
    result.peak_mem_mb = peak_mem_mb()
    return result, outputs

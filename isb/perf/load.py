"""Load-pattern timing (design.md §12.17): how standard inference benchmarks issue requests.

`run_offline` is MLPerf Offline / `vllm bench throughput`: the units go out as one sweep, one call
per chunk, chunks back to back. `run_concurrent` is the closed-loop serving pattern (`vllm bench
serve --max-concurrency`, AIPerf's concurrency sweep): `concurrency` clients each issue one unit
per call and the next as soon as theirs returns, until the units run out. A backend that cannot
take concurrent requests is driven through one lock, the way a serial server queues them; the time
a request waits for the lock counts toward its latency, as a client would see it.

Both run one untimed warm-up (a full pass offline; one request per client concurrently), then
`passes` timed sweeps. A pass is timed between CUDA syncs; a request from issue to return. Call
preparation (the fresh parameter copy) happens outside both. The outputs of the first `keep` units
of the last pass are returned for scoring.
"""
from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from .timing import force_gc, peak_mem_mb, reset_peak_mem, sync_cuda


@dataclass
class LoadResult:
    units: int                       # units per pass
    calls: int                       # calls per pass
    concurrency: int                 # clients (1 offline)
    serial: bool                     # requests were serialized through one lock
    pass_ms: list = field(default_factory=list)
    request_ms: list = field(default_factory=list)
    peak_mem_mb: float = 0.0

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
                "request_mean_ms": statistics.fmean(self.request_ms)}


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
    index -> output for the first `keep` units of the last pass. The first error a client meets
    is raised once every client has stopped."""
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

    sweep(list(range(min(concurrency, len(units)))), timed=False)
    force_gc()
    reset_peak_mem()
    for _ in range(passes):
        outputs = sweep(list(range(len(units))), timed=True)
    result.peak_mem_mb = peak_mem_mb()
    return result, outputs

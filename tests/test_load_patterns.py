"""Load patterns (design.md §12.17): offline and closed-loop concurrent regimes. No GPU — fake
backends and cells.

Pins: the regimes derive from a workload's own regimes (and are reported when a workload cannot
take one); they round-trip through the job contract without changing other regimes' identity;
offline issues chunks back to back and concurrent keeps `concurrency` requests in flight, or one
at a time through a lock on a backend that cannot take concurrent requests; the executor records
the pattern's timing and the first `score_units` outputs, and scoring compares an offline cell
with the reference's per-unit outputs."""
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.sweep.execute as execute_mod  # noqa: E402
from isb.perf.load import run_concurrent, run_offline  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.suites import parse_load  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402
from isb.sweep.spec import (BaselineSpec, CellConfig, Concurrent, ExecutionRegime,  # noqa: E402
                            Offline, instantiate, is_load_regime, reference_key)
from isb.workloads import WORKLOADS  # noqa: E402


# ---- derivation ---------------------------------------------------------------------------------

def test_loads_derive_from_the_workloads_own_regimes():
    spec, skipped = instantiate(WORKLOADS["steering"], "gpt2", size=4,
                                loads=(Offline(24, (8, None)), Concurrent(12, (1, 4))))
    names = [r.name for r in spec.regimes]
    assert names == ["interactive", "batched", "offline/b8", "offline/all",
                     "concurrent/c1", "concurrent/c4"]
    offline = spec.regimes[2]
    assert len(offline.prompts) == 24 and offline.batch_size == 8 and not offline.aggregate
    assert offline.data_name == "counterfact" and offline.passes == 3
    concurrent = spec.regimes[5]
    assert len(concurrent.prompts) == 12 and concurrent.concurrency == 4 and concurrent.passes == 1
    assert not skipped


def test_a_load_the_workload_cannot_take_is_reported():
    spec, skipped = instantiate(WORKLOADS["gen_steering"], "gpt2", size=4,
                                loads=(Offline(16), Concurrent(8, (2,))))
    assert [r.name for r in spec.regimes] == ["generation", "concurrent/c2"]
    assert spec.regimes[1].new_tokens == spec.regimes[0].new_tokens > 0
    assert skipped == [{"label": "[offline load]",
                        "reason": "offline needs a batched regime (this workload's cells take one "
                                  "unit per call)"}]


def test_regimes_filter_keeps_only_the_load_patterns():
    spec, _ = instantiate(WORKLOADS["logit_lens"], "gpt2", size=4, regimes=("offline",),
                          loads=(Offline(16, (4,)), Concurrent(8)))
    assert [r.name for r in spec.regimes] == ["offline/b4"]


def test_regime_names_and_validation():
    assert ExecutionRegime("offline", ["a"], batch_size=None).name == "offline/all"
    assert ExecutionRegime("concurrent", ["a"], concurrency=16).name == "concurrent/c16"
    assert ExecutionRegime("batched", ["a"]).name == "batched"
    assert is_load_regime("offline/b16") and is_load_regime("concurrent/c2")
    assert not is_load_regime("offline_perprompt") and not is_load_regime("offline/b0")
    with pytest.raises(ValueError):
        ExecutionRegime("concurrent", ["a"])
    with pytest.raises(ValueError):
        ExecutionRegime("offline", ["a"], batch_size=0)
    assert reference_key("offline/b4", "r") == ("offline_perprompt", "r")
    assert reference_key("offline/b4", "r", "equivalence") == ("offline/b4", "r")
    assert reference_key("concurrent/c4", "r") == ("concurrent/c4", "r")


def test_parse_load():
    assert parse_load("offline", "256:16,64,all") == Offline(256, (16, 64, None))
    assert parse_load("concurrent", "128:1,4") == Concurrent(128, (1, 4))
    for bad in ("256", "0:4", "256:x", "256:"):
        with pytest.raises(ValueError):
            parse_load("offline", bad)
    with pytest.raises(ValueError):
        parse_load("concurrent", "8:all")


def test_load_regimes_round_trip_the_job_contract(tmp_path):
    from isb.jobs.contract import expected_cells, prepare, read_experiment, restore_spec, unpack

    plain, _ = instantiate(WORKLOADS["steering"], "gpt2", size=4)
    loaded, _ = instantiate(WORKLOADS["steering"], "gpt2", size=4,
                            loads=(Offline(8, (4, None)), Concurrent(6, (3,))))
    experiment = prepare(loaded, tmp_path / "loaded")
    record = unpack(read_experiment(tmp_path / "loaded")["spec"])
    by_kind = {r["kind"]: r for r in record["regimes"]}
    assert "batch_size" not in by_kind["interactive"] and "passes" not in by_kind["batched"]
    restored, _ = restore_spec(tmp_path / "loaded")
    assert [r.name for r in restored.regimes] == [r.name for r in loaded.regimes]
    assert restored.regimes[3].batch_size is None and restored.regimes[4].concurrency == 3
    cells = expected_cells(experiment)
    assert ("offline/b4", plain.tasks[0].label) in cells
    assert ("concurrent/c3", plain.tasks[0].label) in cells
    # A spec without load patterns freezes to the record it always did.
    prepare(plain, tmp_path / "plain")
    plain_record = unpack(read_experiment(tmp_path / "plain")["spec"])
    assert all(set(r) == {"kind", "units", "new_tokens", "aggregate", "data_knobs", "data_name"}
               for r in plain_record["regimes"])


# ---- timing loops -------------------------------------------------------------------------------

class _InFlight:
    """Counts calls in progress; each call sleeps briefly so overlapping clients would show."""
    def __init__(self):
        self.lock, self.now, self.peak, self.calls = threading.Lock(), 0, 0, []

    def call(self, unit):
        def invoke():
            with self.lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
                self.calls.append(unit)
            time.sleep(0.01)
            with self.lock:
                self.now -= 1
            return torch.tensor([[float(unit)]])
        return invoke


def test_offline_issues_chunks_back_to_back_and_keeps_the_first_units():
    seen = []

    def prepare(chunk):
        return lambda: seen.append(list(chunk)) or torch.tensor([[float(u)] for u in chunk])
    chunks = [[0, 1, 2], [3, 4, 5], [6, 7]]
    result, outputs = run_offline(prepare, chunks, passes=2, keep=4)
    assert seen == chunks * 3                       # one warm-up pass, then two timed passes
    assert result.units == 8 and result.calls == 3 and len(result.pass_ms) == 2
    assert len(result.request_ms) == 6 and sorted(outputs) == [0, 1]
    assert result.throughput > 0 and result.summary()["passes"] == 2


def test_concurrent_keeps_c_requests_in_flight():
    probe = _InFlight()
    result, outputs = run_concurrent(probe.call, list(range(12)), concurrency=4, passes=1,
                                     keep=5, serial=False)
    assert probe.peak == 4 and result.concurrency == 4 and not result.serial
    assert len(result.request_ms) == 12 and len(probe.calls) == 4 + 12   # warm-up, then the pass
    assert sorted(outputs) == [0, 1, 2, 3, 4]


def test_serial_backend_takes_one_request_at_a_time_and_queues_the_rest():
    probe = _InFlight()
    result, _ = run_concurrent(probe.call, list(range(8)), concurrency=4, passes=1, keep=1,
                               serial=True)
    assert probe.peak == 1 and result.serial
    # Queueing for the lock counts toward latency: the slowest request waited for the others.
    assert result.percentile(99) > 2 * result.percentile(0)


def test_a_client_error_stops_the_sweep_and_is_raised():
    def prepare(unit):
        def invoke():
            if unit == 3:
                raise RuntimeError("unit 3 failed")
            return torch.zeros(1, 1)
        return invoke
    with pytest.raises(RuntimeError, match="unit 3 failed"):
        run_concurrent(prepare, list(range(6)), concurrency=2, passes=1, keep=1, serial=False)


# ---- executor and scoring -----------------------------------------------------------------------

def _onehot(i, vocab=8):
    t = torch.zeros(1, vocab)
    t[0, i] = 12.0
    return t


class _Backend:
    name = "hf"

    def __init__(self, concurrent=False):
        self.many, self.plain, self.sessions = [], [], 0
        self.in_flight = _InFlight()
        if concurrent:
            self.concurrent_requests = True

    def load(self, repo):
        return object()

    def teardown(self, model):
        pass

    def last(self, t):
        return t

    def vanilla(self, model, prompt, *, new_tokens):
        self.plain.append(prompt)

    def vanilla_many(self, model, prompts, *, new_tokens):
        self.many.append(list(prompts))

    def concurrent_session(self):
        import contextlib

        backend = self

        @contextlib.contextmanager
        def session():
            backend.sessions += 1
            yield
        return session()


def _cells(wrong_on=None):
    def fake(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            impl.in_flight.call(0)()
            outs = [_onehot(7) if p == wrong_on else _onehot(int(p[1])) for p in prompts]
            return torch.cat(outs, dim=0)
        return fn
    return fake


def _spec():
    units = [f"p{i}" for i in range(6)]
    return CellConfig(
        name="load", methodology="m", family="f", repo="r",
        protocol_absence_reason="Synthetic load-pattern fixture",
        regimes=[ExecutionRegime("offline", units, batch_size=4, score_units=3, passes=2),
                 ExecutionRegime("concurrent", units, concurrency=3, score_units=4)],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)


def _execute(tmp, name, engine_kind, backend, wrong_on=None):
    saved = execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance
    execute_mod.get_cell = _cells(wrong_on)
    execute_mod.make_backend = lambda run, spec: backend
    execute_mod.resolve_provenance = lambda run: {
        "client": {"nnsight": {"commit": f"c-{name}"}}, "deployment": {"kind": "local"},
        "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": {}},
        "host": {"hostname": "testbox", "gpus": []}}
    try:
        execute_run(_spec(), RunConfig(engine=EngineConfig(engine_kind)), str(tmp), name)
    finally:
        execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance = saved


def test_executor_records_load_patterns(tmp_path):
    from isb.runfile import load_run

    backend = _Backend()
    _execute(tmp_path, "ref", "transformers", backend)
    out, prov = load_run(str(tmp_path), "ref")
    meta = out[("__meta__",)]
    offline, concurrent = meta[("offline/b4", "t")], meta[("concurrent/c3", "t")]
    assert offline["error"] is None and offline["load"]["calls_per_pass"] == 2
    assert offline["load"]["passes"] == 2 and offline["throughput"] > 0
    assert concurrent["load"]["concurrency"] == 3 and concurrent["load"]["serial"]
    assert backend.in_flight.peak == 1                  # no concurrent_requests -> serialized
    # The first score_units outputs are kept; offline also records its per-unit reference.
    assert out[("offline/b4", "t")].shape == (3, 8)
    assert out[("concurrent/c3", "t")].shape == (4, 8)
    assert out[("offline_perprompt", "t")].shape == (3, 8)
    # The plain system runs under the same pattern.
    assert meta[("__vanilla__", "offline/b4")]["error"] is None
    assert backend.many[:2] == [["p0", "p1", "p2", "p3"], ["p4", "p5"]]
    assert meta[("__baseline__", "concurrent/c3")]["load"]["units"] == 6
    assert offline["overhead_vs_vanilla"] is not None
    loads = [r["load"] for r in prov["coordinates"]["regimes"]]
    assert loads[0]["batch_size"] == 4 and loads[1]["concurrency"] == 3


def test_a_concurrent_backend_opens_its_session_and_overlaps_requests(tmp_path):
    backend = _Backend(concurrent=True)
    _execute(tmp_path, "cand", "vllm", backend)
    assert backend.sessions == 3                       # baseline, plain system, one case
    assert backend.in_flight.peak > 1


def test_offline_candidate_scored_against_reference_per_unit_outputs(tmp_path):
    _execute(tmp_path, "ref", "transformers", _Backend())
    _execute(tmp_path, "cand", "vllm", _Backend(concurrent=True), wrong_on="p1")
    by = {(c.workload, c.label): c for c in score_runs(_spec(), str(tmp_path), "cand", "ref",
                                                        quiet=True)}
    assert set(by) == {("offline/b4", "t"), ("concurrent/c3", "t")}
    assert by[("offline/b4", "t")].state == "NUMERICAL_MISMATCH"     # p1 is among the scored units
    assert by[("concurrent/c3", "t")].state == "NUMERICAL_MISMATCH"

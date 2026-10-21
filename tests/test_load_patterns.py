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
from isb.perf.load import arrival_offsets, run_concurrent, run_offline, run_open  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.suites import Entry, cases, parse_load  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402
from isb.sweep.spec import (BaselineSpec, CellConfig, Concurrent, ExecutionRegime,  # noqa: E402
                            Offline, Open, instantiate, is_load_regime, reference_key)
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


def test_a_load_the_workload_does_not_declare_derives_nothing():
    spec, skipped = instantiate(WORKLOADS["ablation"], "gpt2", size=4,
                                loads=(Offline(16, (4,)), Concurrent(8, (2,)), Open(8, (4,))))
    assert [r.name for r in spec.regimes] == ["interactive", "batched", "offline/b4"]
    assert [row["label"] for row in skipped if row["label"].endswith(" load]")] == \
        ["[concurrent load]", "[open load]"]
    assert all("does not match how this workload is used (its load patterns: offline)" in row["reason"]
               for row in skipped if row["label"].endswith(" load]"))


def test_single_unit_cells_sweep_offline_at_b1_and_serve_from_their_single_unit_regime():
    spec, skipped = instantiate(WORKLOADS["gen_steering"], "gpt2", size=4,
                                loads=(Offline(16, (1, 8)), Concurrent(8, (2,)), Open(8, (2.5,))))
    assert [r.name for r in spec.regimes] == ["generation", "offline/b1", "concurrent/c2", "open/r2.5"]
    assert all(r.new_tokens == spec.regimes[0].new_tokens > 0 for r in spec.regimes)
    assert len(spec.regimes[1].prompts) == 16 and len(spec.regimes[3].prompts) == 8
    assert skipped == [{"label": "[offline load]",
                        "reason": "this workload's cells take one unit per call, so offline runs "
                                  "at B=1 only"}]


def test_a_whole_list_cell_that_declares_offline_sweeps_one_unit_per_call():
    # Jacobian collection's one regime hands every prompt to one call; declaring offline says the
    # per-prompt work is independent, so the sweep issues one prompt per call and nothing larger
    spec, skipped = instantiate(WORKLOADS["jacobian_collect"], "gpt2", size=4,
                                loads=(Offline(8, (1, 4)),))
    assert [r.name for r in spec.regimes] == ["interactive", "offline/b1"]
    assert len(spec.regimes[1].prompts) == 8 and not spec.regimes[0].aggregate
    assert [row["reason"] for row in skipped] == \
        ["this workload's cells take one unit per call, so offline runs at B=1 only"]
    # DAS has the same whole-list regime but declares nothing: a fit's steps depend on each other
    spec, skipped = instantiate(WORKLOADS["das"], "gpt2", loads=(Offline(8, (1,)),))
    assert [r.name for r in spec.regimes] == ["interactive"]
    assert "its load patterns: none" in skipped[-1]["reason"]


def test_regimes_filter_keeps_only_the_load_patterns():
    spec, _ = instantiate(WORKLOADS["logit_lens"], "gpt2", size=4, regimes=("offline",),
                          loads=(Offline(16, (4,)), Concurrent(8)))
    assert [r.name for r in spec.regimes] == ["offline/b4"]


def test_a_workload_left_with_no_regime_is_skipped_with_the_load_reason():
    specs, skipped = cases([Entry(("steering", "gen_steering"), ("gpt2",), size=4,
                                         regimes=("offline",), loads=(Offline(16, (4,)),))])
    assert [s.name for s in specs] == ["steering/gpt2"]
    [whole] = [row for row in skipped if row["label"] == "*"]
    assert whole["case"] == "gen_steering/gpt2" and "offline runs at B=1 only" in whole["reason"]


def test_regime_names_and_validation():
    assert ExecutionRegime("offline", ["a"], batch_size=None).name == "offline/all"
    assert ExecutionRegime("concurrent", ["a"], concurrency=16).name == "concurrent/c16"
    assert ExecutionRegime("batched", ["a"]).name == "batched"
    assert ExecutionRegime("open", ["a"], rate=2.5).name == "open/r2.5"
    assert ExecutionRegime("open", ["a"], rate=16.0).name == "open/r16"
    assert is_load_regime("offline/b16") and is_load_regime("concurrent/c2")
    assert is_load_regime("open/r2.5") and is_load_regime("open/r16")
    assert not is_load_regime("offline_perprompt") and not is_load_regime("offline/b0")
    assert not is_load_regime("open/r") and not is_load_regime("open/r1e-07")
    for rate in (0.0, -1.0, 1e-7):
        with pytest.raises(ValueError):
            ExecutionRegime("open", ["a"], rate=rate)
    with pytest.raises(ValueError):
        ExecutionRegime("concurrent", ["a"])
    with pytest.raises(ValueError):
        ExecutionRegime("offline", ["a"], batch_size=0)
    assert reference_key("offline/b4", "r") == ("offline_perprompt", "r")
    assert reference_key("offline/b4", "r", "equivalence") == ("offline/b4", "r")
    assert reference_key("concurrent/c4", "r") == ("concurrent/c4", "r")
    assert reference_key("open/r4", "r") == ("open/r4", "r")


def test_parse_load():
    assert parse_load("offline", "256:16,64,all") == Offline(256, (16, 64, None))
    assert parse_load("concurrent", "128:1,4") == Concurrent(128, (1, 4))
    for bad in ("256", "0:4", "256:x", "256:"):
        with pytest.raises(ValueError):
            parse_load("offline", bad)
    with pytest.raises(ValueError):
        parse_load("concurrent", "8:all")
    assert parse_load("open", "128:4,2.5") == Open(128, (4.0, 2.5))
    for bad in ("128:0", "128:all", "128:x", "128:1e3"):
        with pytest.raises(ValueError):
            parse_load("open", bad)


def test_load_regimes_round_trip_the_job_contract(tmp_path):
    from isb.jobs.contract import expected_cells, prepare, read_experiment, restore_spec, unpack

    plain, _ = instantiate(WORKLOADS["steering"], "gpt2", size=4)
    loaded, _ = instantiate(WORKLOADS["steering"], "gpt2", size=4,
                            loads=(Offline(8, (4, None)), Concurrent(6, (3,)), Open(5, (2.5,))))
    experiment = prepare(loaded, tmp_path / "loaded")
    record = unpack(read_experiment(tmp_path / "loaded")["spec"])
    by_kind = {r["kind"]: r for r in record["regimes"]}
    assert "batch_size" not in by_kind["interactive"] and "passes" not in by_kind["batched"]
    # Each load pattern records only its own settings, so offline and concurrent records are the
    # ones they were before the open loop existed.
    assert "rate" not in by_kind["offline"] and "rate" not in by_kind["concurrent"]
    assert by_kind["open"]["rate"] == 2.5 and "concurrency" not in by_kind["open"]
    restored, _ = restore_spec(tmp_path / "loaded")
    assert [r.name for r in restored.regimes] == [r.name for r in loaded.regimes]
    assert restored.regimes[3].batch_size is None and restored.regimes[4].concurrency == 3
    assert restored.regimes[5].rate == 2.5
    cells = expected_cells(experiment)
    assert ("offline/b4", plain.tasks[0].label) in cells
    assert ("concurrent/c3", plain.tasks[0].label) in cells
    assert ("open/r2.5", plain.tasks[0].label) in cells
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


def test_concurrent_pass_time_excludes_call_preparation():
    def prepare(unit):
        time.sleep(0.02)
        return lambda: None
    result, _ = run_concurrent(prepare, list(range(8)), concurrency=1, passes=1, keep=0,
                               serial=True)
    assert result.pass_ms[0] < 8 * 20 / 2      # eight preparations would add 160 ms


def test_a_client_error_stops_the_sweep_and_is_raised():
    def prepare(unit):
        def invoke():
            if unit == 3:
                raise RuntimeError("unit 3 failed")
            return torch.zeros(1, 1)
        return invoke
    with pytest.raises(RuntimeError, match="unit 3 failed"):
        run_concurrent(prepare, list(range(6)), concurrency=2, passes=1, keep=1, serial=False)


def test_open_loop_arrivals_are_a_seeded_poisson_process():
    offsets = arrival_offsets(4000, 50.0)
    assert offsets == arrival_offsets(4000, 50.0) and offsets[0] == 0.0
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert all(g > 0 for g in gaps) and abs(sum(gaps) / len(gaps) - 1 / 50.0) < 0.1 / 50.0


def test_open_loop_issues_on_arrival_without_waiting_for_returns():
    probe = _InFlight()                                 # each call takes 10 ms
    result, outputs = run_open(probe.call, list(range(12)), rate=2000.0, passes=1, keep=3,
                               serial=False)
    assert probe.peak > 1 and result.max_in_flight > 1 and result.concurrency == 0
    assert len(result.request_ms) == 12 and len(probe.calls) == 1 + 12   # warm-up, then the pass
    assert sorted(outputs) == [0, 1, 2]
    assert result.summary()["offered_rate"] == 2000.0


def test_open_loop_on_a_serial_backend_queues_and_latency_counts_from_arrival():
    probe = _InFlight()
    # Arrivals at 1000/s against a 10 ms serial service: every request waits for those before it.
    result, _ = run_open(probe.call, list(range(8)), rate=1000.0, passes=1, keep=0, serial=True)
    assert probe.peak == 1 and result.max_in_flight > 1
    assert result.percentile(99) > 50.0 > result.percentile(0)


def test_open_loop_below_capacity_keeps_latency_near_the_service_time():
    probe = _InFlight()
    result, _ = run_open(probe.call, list(range(6)), rate=20.0, passes=1, keep=0, serial=True)
    assert result.percentile(99) < 40.0                 # 10 ms calls, arrivals ~50 ms apart


def test_an_open_loop_error_is_raised():
    def prepare(unit):
        def invoke():
            if unit == 2:
                raise RuntimeError("unit 2 failed")
            return torch.zeros(1, 1)
        return invoke
    with pytest.raises(RuntimeError, match="unit 2 failed"):
        run_open(prepare, list(range(5)), rate=500.0, passes=1, keep=1, serial=False)


# ---- executor and scoring -----------------------------------------------------------------------

def _onehot(i, vocab=8):
    t = torch.zeros(1, vocab)
    t[0, i] = 12.0
    return t


class _Backend:
    name = "hf"

    def __init__(self, concurrent=False):
        self.many, self.plain, self.sessions, self.calls = [], [], 0, []
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


def _cells(wrong_on=None, rows=1):
    def fake(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            impl.calls.append(list(prompts))
            impl.in_flight.call(0)()
            outs = [_onehot(7) if p == wrong_on else _onehot(int(p[1])) for p in prompts]
            return torch.cat([o for o in outs for _ in range(rows)], dim=0)
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


def _execute(tmp, name, engine_kind, backend, wrong_on=None, spec=None, rows=1):
    saved = execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance
    execute_mod.get_cell = _cells(wrong_on, rows)
    execute_mod.make_backend = lambda run, spec: backend
    execute_mod.resolve_provenance = lambda run: {
        "client": {"nnsight": {"commit": f"c-{name}"}}, "deployment": {"kind": "local"},
        "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": {}},
        "host": {"hostname": "testbox", "gpus": []}}
    try:
        execute_run(spec or _spec(), RunConfig(engine=EngineConfig(engine_kind)), str(tmp), name)
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


def _serving_spec():
    pairs = [(f"p{i}", f"q{i}") for i in range(4)]
    return CellConfig(
        name="serving", methodology="m", family="f", repo="r",
        protocol_absence_reason="Synthetic load-pattern fixture",
        regimes=[ExecutionRegime("offline", pairs, batch_size=1, score_units=2),
                 ExecutionRegime("open", [f"p{i}" for i in range(6)], rate=1000.0, score_units=3)],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)


def test_executor_runs_single_unit_offline_and_the_open_loop(tmp_path):
    from isb.runfile import load_run

    backend = _Backend(concurrent=True)
    _execute(tmp_path, "cand", "vllm", backend, spec=_serving_spec())
    out, _ = load_run(str(tmp_path), "cand")
    meta = out[("__meta__",)]
    # Offline at B=1 hands a single-unit cell each unit's prompt list, a pair as [clean, corrupted].
    assert ["p0", "q0"] in backend.calls and [("p0", "q0")] not in backend.calls
    assert meta[("offline/b1", "t")]["load"]["calls_per_pass"] == 4
    assert out[("offline/b1", "t")].shape == (4, 8)     # two units, two rows each
    opened = meta[("open/r1000", "t")]
    assert opened["error"] is None and opened["load"]["offered_rate"] == 1000.0
    assert out[("open/r1000", "t")].shape == (3, 8)
    assert backend.sessions == 3                       # the open loop runs in a concurrent session


def test_offline_keeps_the_rows_its_scored_units_fill(tmp_path):
    """A unit may fill several rows (decode steps, tokens): offline/b4 with three scored units of
    two rows each keeps six rows, the per-unit reference's, not three."""
    from isb.runfile import load_run

    _execute(tmp_path, "ref", "transformers", _Backend(), rows=2)
    out, _ = load_run(str(tmp_path), "ref")
    assert out[("offline_perprompt", "t")].shape == (6, 8)
    assert torch.equal(out[("offline/b4", "t")], out[("offline_perprompt", "t")])


def test_offline_candidate_scored_against_reference_per_unit_outputs(tmp_path):
    _execute(tmp_path, "ref", "transformers", _Backend())
    _execute(tmp_path, "cand", "vllm", _Backend(concurrent=True), wrong_on="p1")
    by = {(c.workload, c.label): c for c in score_runs(_spec(), str(tmp_path), "cand", "ref",
                                                        quiet=True)}
    assert set(by) == {("offline/b4", "t"), ("concurrent/c3", "t")}
    assert by[("offline/b4", "t")].state == "NUMERICAL_MISMATCH"     # p1 is among the scored units
    assert by[("concurrent/c3", "t")].state == "NUMERICAL_MISMATCH"


def test_coordinates_rebuilt_from_a_frozen_record_keep_load_regimes_apart(tmp_path):
    """The manager rebuilds submitted coordinates from the experiment record, whose load regimes
    carry their settings as flat fields; two offline regimes must stay distinct regimes."""
    from isb.jobs.contract import prepare, read_experiment, wire_spec
    from isb.runs import run_coordinates

    spec, _ = instantiate(WORKLOADS["steering"], "gpt2", size=4, regimes=("offline",),
                          loads=(Offline(8, (4, None)),))
    experiment = prepare(spec, tmp_path / "exp")
    record = wire_spec(read_experiment(tmp_path / "exp"))
    coordinates = run_coordinates(
        spec=record["name"], methodology=record["methodology"], family=record["family"],
        repo=record["repo"], interface="hf",
        regimes=[{**w, "data": w.get("data_name")} for w in record["regimes"]],
        cases=record["tasks"], inputs_sha256=experiment["inputs_sha256"])
    assert [r["load"]["batch_size"] for r in coordinates["regimes"]] == [4, None]

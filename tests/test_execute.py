"""Executor tests (isb/sweep/execute.py), no GPU: fake backends and cells. Execution writes one
run file holding every cell's check output, the per-unit stack a batched call is scored against,
and provenance with the run's coordinates; scored outputs come from check calls, never timed
ones."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.sweep.execute as execute_mod  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime  # noqa: E402


def _onehot(i, vocab=8):
    t = torch.zeros(1, vocab)
    t[0, i] = 12.0
    return t


class _FakeBackend:
    def between_calls(self, model):
        pass

    def load(self, repo):
        return object()

    def teardown(self, model):
        pass

    def last(self, t):
        return t


def _spec():
    return CellConfig(
        name="xs", methodology="m", family="f", repo="r",
        protocol_absence_reason="Synthetic execution and scoring fixture",
        regimes=[ExecutionRegime("interactive", ["p0", "p1"]),
                   ExecutionRegime("batched", ["p0", "p1"])],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None,
        warmup=0, n_trials=1)


def _fake_cells(methodology, family, backend):
    """Cell registry stub: one onehot per prompt."""
    def fn(impl, model, prompts, **params):
        return torch.cat([_onehot(int(p[1])) for p in prompts], dim=0)
    return fn


def _execute(tmp, name, engine_kind):
    run = RunConfig(engine=EngineConfig(engine_kind))
    orig_gc = execute_mod.get_cell
    orig_mb, orig_prov = execute_mod.make_backend, execute_mod.resolve_provenance
    execute_mod.get_cell = _fake_cells
    execute_mod.make_backend = lambda run, spec: _FakeBackend()
    execute_mod.resolve_provenance = lambda run: {
        "client": {"nnsight": {"commit": f"c-{name}"}, "vllm": None, "transformers": "5.x"},
        "deployment": {"kind": "local"},
        "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": {}},
        "host": {"hostname": "testbox", "gpus": [{"name": "FakeGPU-80GB", "vram_gb": 80}]},
    }
    try:
        execute_run(_spec(), run, str(tmp), name)
    finally:
        execute_mod.get_cell = orig_gc
        execute_mod.make_backend, execute_mod.resolve_provenance = orig_mb, orig_prov


def test_execute_writes_one_selfcontained_run_file(tmp_path):
    from isb.runfile import load_run
    _execute(tmp_path, "ref", "transformers")
    assert (tmp_path / "ref.pt").exists() and not (tmp_path / "ref.json").exists()
    out, prov = load_run(str(tmp_path), "ref")
    assert ("interactive", "t") in out and ("batched", "t") in out
    assert ("batched_perprompt", "t") in out          # any run can reference a padded compare
    meta = out[("__meta__",)]
    assert meta[("interactive", "t")]["error"] is None
    assert meta[("interactive", "t")]["median_latency_ms"] is not None
    coords = prov["coordinates"]
    assert coords["spec"] == "xs" and coords["interface"] == "hf"
    assert {w["kind"] for w in coords["regimes"]} == {"interactive", "batched"}


def _checked_cells(calls):
    """Cells that declare `check`: a timed call returns -1 everywhere, a result no check reads; a
    check call returns the per-prompt onehots. `calls` records each call's kind."""
    def fake(methodology, family, backend):
        def fn(impl, model, prompts, *, check=False):
            calls.append(check)
            if not check:
                return torch.full((len(prompts), 8), -1.0)
            return torch.cat([_onehot(int(p[1])) for p in prompts], dim=0)
        return fn
    return fake


def _run(tmp, monkeypatch, spec, calls, met=None):
    from isb.oracle import checks
    from isb.runfile import load_run

    def meets(timed, checked):
        (met if met is not None else []).append((timed, checked))
        return True, {}

    monkeypatch.setattr(checks, "TIMED", {"m": meets})
    monkeypatch.setattr(execute_mod, "get_cell", _checked_cells(calls))
    monkeypatch.setattr(execute_mod, "make_backend", lambda run, spec: _FakeBackend())
    monkeypatch.setattr(execute_mod, "resolve_provenance", lambda run: {
        "client": {}, "deployment": {"kind": "local"},
        "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": {}},
        "host": {"hostname": "testbox", "gpus": []}})
    execute_run(spec, RunConfig(engine=EngineConfig("transformers")), str(tmp), "run")
    return load_run(str(tmp), "run")[0]


def test_scored_outputs_come_from_check_calls_never_timed_ones(tmp_path, monkeypatch):
    calls = []
    out = _run(tmp_path, monkeypatch, _spec(), calls)
    expected = torch.cat([_onehot(0), _onehot(1)])
    for key in (("interactive", "t"), ("batched", "t"), ("batched_perprompt", "t")):
        assert torch.equal(out[key], expected), key
    assert False in calls and True in calls
    assert out[("__meta__",)][("batched", "t")]["median_latency_ms"] is not None


def test_check_calls_follow_every_timed_call_and_meet_its_result(tmp_path, monkeypatch):
    calls, met = [], []
    spec = CellConfig(name="xs", methodology="m", family="f", repo="r",
                      protocol_absence_reason="Synthetic execution and scoring fixture",
                      regimes=[ExecutionRegime("batched", ["p0", "p1"])],
                      tasks=[({}, "t"), ({}, "u")], baseline=BaselineSpec(params={}),
                      effect=None, warmup=0, n_trials=1)
    out = _run(tmp_path, monkeypatch, spec, calls, met)
    # baseline, t and u timed; then the per-unit references (two units each) and t, u checked
    assert calls == [False] * 3 + [True] * 6
    onehots = torch.cat([_onehot(0), _onehot(1)])
    assert len(met) == 2 and all(torch.equal(timed, torch.full((2, 8), -1.0))
                                 and torch.equal(checked, onehots) for timed, checked in met)
    record = out[("__meta__",)][("batched", "t")]
    assert record["timed_output"] == "result" and record["timed_check"] == {"passed": True}


def test_a_check_run_makes_the_check_calls_only(tmp_path, monkeypatch):
    calls, spec = [], _spec()
    spec.warmup = spec.n_trials = 0
    out = _run(tmp_path, monkeypatch, spec, calls)
    meta = out[("__meta__",)]
    assert calls and all(calls)
    assert not any(key[0] in ("__baseline__", "__vanilla__") for key in meta)
    assert meta[("batched", "t")]["error"] is None
    assert "median_latency_ms" not in meta[("batched", "t")]
    assert torch.equal(out[("batched", "t")], torch.cat([_onehot(0), _onehot(1)]))


def _run_all():
    print("uses pytest fixtures (tmp_path/capsys); run via pytest")


if __name__ == "__main__":
    _run_all()

"""Execute/score layer tests (isb/sweep/execute.py, score.py); no GPU — fake backends + cells.

Pins the decoupling contract: execution ALWAYS writes outputs (own batched output AND the
per-prompt reference stack) plus provenance with the run's coordinates; scoring is a pure step
over the files that derives its axis from provenance (cross-engine -> correctness, same engine ->
equivalence), scores batched candidates against the reference's per-prompt stack, and applies the
equivalence), and scores batched candidates against the reference's per-prompt stack."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.sweep.execute as execute_mod  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime  # noqa: E402


def _onehot(i, vocab=8):
    t = torch.zeros(1, vocab)
    t[0, i] = 12.0
    return t


class _FakeBackend:
    def load(self, repo):
        return object()

    def teardown(self, model):
        pass

    def last(self, t):
        return t


def _spec():
    return CellConfig(
        name="xs", methodology="m", family="f", repo="r",
        regimes=[ExecutionRegime("interactive", ["p0", "p1"]),
                   ExecutionRegime("batched", ["p0", "p1"])],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None,
        warmup=0, n_trials=1)


def _fake_cells(wrong_on=None):
    """Cell registry stub: per-prompt onehots; `wrong_on` makes that prompt's output wrong."""
    def fake(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            outs = []
            for p in prompts:
                i = int(p[1])
                outs.append(_onehot(7) if p == wrong_on else _onehot(i))
            return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        return fn
    return fake


def _execute(tmp, name, engine_kind, wrong_on=None):
    run = RunConfig(engine=EngineConfig(engine_kind))
    orig_gc = execute_mod.get_cell
    orig_mb, orig_prov = execute_mod.make_backend, execute_mod.resolve_provenance
    execute_mod.get_cell = _fake_cells(wrong_on)
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


def test_errored_cells_appear_as_error_rows(tmp_path):
    """An errored cell writes no output tensor; scoring must still produce its ERROR row."""
    import isb.sweep.execute as em

    _execute(tmp_path, "ref", "transformers")
    orig = execute_mod.get_cell

    def raising(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            raise RuntimeError("guarded")
        return fn
    execute_mod.get_cell = raising
    em.make_backend = lambda run, spec: _FakeBackend()
    em.resolve_provenance = lambda run: {
        "client": {"nnsight": {"commit": "c-err"}}, "deployment": {"kind": "local"},
        "engine": {"kind": "vllm", "mode": "async", "params": {}},
        "host": {"hostname": "testbox", "gpus": []},
    }
    try:
        execute_run(_spec(), RunConfig(engine=EngineConfig("vllm")), str(tmp_path), "errcand")
    finally:
        execute_mod.get_cell = orig
    cells = score_runs(_spec(), str(tmp_path), "errcand", "ref", quiet=True)
    assert cells and all(c.state == "ERROR" for c in cells)
    assert "guarded" in cells[0].error


def test_score_cross_engine_is_correctness_and_catches_the_wrong_prompt(tmp_path, capsys):
    _execute(tmp_path, "ref", "transformers")
    _execute(tmp_path, "cand", "vllm", wrong_on="p1")   # wrong on one of two prompts
    results = score_runs(_spec(), str(tmp_path), "cand", "ref")
    by = {(c.workload, c.label): c for c in results}
    cell = by[("interactive", "t")]
    assert cell.state == "SILENTLY_WRONG"               # 1/2 top-1 agreement -> caught
    header = capsys.readouterr().out
    assert "correctness" in header and "c-cand" in header and "c-ref" in header


def test_score_same_engine_is_equivalence_axis(tmp_path):
    _execute(tmp_path, "one", "vllm")
    _execute(tmp_path, "two", "vllm")
    results = score_runs(_spec(), str(tmp_path), "two", "one")
    assert all(c.state in ("EQUIVALENT", "NO_REFERENCE") for c in results)


def test_batched_candidate_scored_against_reference_perprompt_stack(tmp_path):
    _execute(tmp_path, "ref", "transformers")
    _execute(tmp_path, "cand", "vllm")
    results = score_runs(_spec(), str(tmp_path), "cand", "ref")
    by = {(c.workload, c.label): c for c in results}
    assert by[("batched", "t")].state == "SUPPORTED"    # identical outputs -> supported


def _run_all():
    print("uses pytest fixtures (tmp_path/capsys); run via pytest")


if __name__ == "__main__":
    _run_all()

"""Sweep-layer invariant tests (isb/sweep/execute.py + score.py) — no GPU; fake backends + cells.

Verifies the harness invariants across the execute/score split: ONE model load per run amortized
across tasks (not per cell); an intervention error is isolated (the engine survives, later tasks
still run, no reload) and still appears as an ERROR row when scored; perf meta is populated for
the cells that ran; batched candidates are scored against the reference's per-prompt stack (a
padded batch is not its own valid reference); a control-dtype run file disambiguates precision
near-ties; aggregate regimes score the verdict over ALL prompts (and pairs).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.sweep.execute as execute_mod  # noqa: E402
from isb.runfile import load_run  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.states import AppState  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime  # noqa: E402

V = 8


def _onehot(i):
    t = torch.zeros(1, V); t[0, i] = 9.0; return t


class _FakeBackend:
    def __init__(self):
        self.n_load = 0
        self.n_teardown = 0

    def load(self, repo):
        self.n_load += 1
        return "model"

    def teardown(self, model):
        self.n_teardown += 1


def _prov(engine_kind):
    return {
        "client": {"nnsight": {"commit": f"c-{engine_kind}"}, "vllm": None, "transformers": "5.x"},
        "deployment": {"kind": "local"},
        "engine": {"kind": engine_kind, "mode": "async", "params": {}},
        "host": {"hostname": "testbox", "gpus": []},
    }


def _execute(tmp, name, engine_kind, spec, get_cell):
    """execute_run with a fake cell registry, backend, and provenance; returns the backend so
    tests can assert load/teardown counts."""
    be = _FakeBackend()
    orig = execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance
    execute_mod.get_cell = get_cell
    execute_mod.make_backend = lambda run, spec: be
    execute_mod.resolve_provenance = lambda run: _prov(run.engine.kind)
    try:
        execute_run(spec, RunConfig(engine=EngineConfig(engine_kind)), str(tmp), name)
    finally:
        execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance = orig
    return be


# distinct outputs per task key; the vllm cell matches hf for "b" and raises for "a"
OUT = {"base": _onehot(0), "a": _onehot(3), "b": _onehot(5)}


def _fake_get_cell(methodology, family, backend):
    def fn(impl, model, prompts, **params):
        k = params.get("k")
        if backend == "vllm_async" and k == "a":
            raise RuntimeError("boom-a")        # an isolated intervention error
        return OUT[k].clone()
    return fn


def _spec():
    return CellConfig(
        name="fake", methodology="m", family="fam", repo="repo://x",
        protocol_absence_reason="Synthetic execution harness fixture",
        regimes=[ExecutionRegime("interactive", ["one prompt"])],
        tasks=[({"k": "a"}, "a"), ({"k": "b"}, "b")],   # task "a" errors on vLLM, "b" succeeds AFTER it
        baseline=BaselineSpec(params={"k": "base"}),
        effect=None, warmup=0, n_trials=1,
    )


def test_mutating_cell_cannot_change_later_calls_or_saved_coordinates(tmp_path, monkeypatch):
    from isb.perf import timing
    from isb.sweep.spec import EffectSpec

    monkeypatch.setattr(timing, "force_gc", lambda: None)
    spec = _spec()
    spec.warmup, spec.n_trials = 2, 3
    spec.regimes = [ExecutionRegime("interactive", ["a", "b"],
                                    data_knobs={"nested": {"values": [1]}}),
                    ExecutionRegime("batched", ["a", "b"],
                                    data_knobs={"nested": {"values": [1]}})]
    spec.effect = EffectSpec({"k": "base"}, {"k": "a"})
    for params in (spec.baseline.params, spec.effect.baseline_params,
                   spec.effect.perturbed_params, spec.tasks[0].semantics):
        params["nested"] = {"values": [1]}
    seen = []

    def registry(*args):
        def cell(impl, model, prompts, **params):
            seen.append(list(params["nested"]["values"]))
            params["nested"]["values"].append(99)
            return _onehot(0)
        return cell

    _execute(tmp_path, "owned", "transformers", spec, registry)
    assert len(seen) > 20
    assert all(values == [1] for values in seen)
    assert all(w.data_knobs["nested"]["values"] == [1] for w in spec.regimes)
    assert spec.baseline.params["nested"]["values"] == [1]
    assert spec.effect.baseline_params["nested"]["values"] == [1]
    assert spec.effect.perturbed_params["nested"]["values"] == [1]
    assert spec.tasks[0].params["nested"]["values"] == [1]
    _, provenance = load_run(str(tmp_path), "owned")
    assert all(w["data_knobs"]["nested"]["values"] == [1]
               for w in provenance["coordinates"]["regimes"])
    assert all(c["semantics"]["nested"]["values"] == [1]
               for w in provenance["coordinates"]["regimes"] for c in w["cases"])


def test_model_loaded_once_per_run_not_per_cell(tmp_path):
    be = _execute(tmp_path, "ref", "transformers", _spec(), _fake_get_cell)
    assert be.n_load == 1                       # amortized across both tasks, not 2 loads
    assert be.n_teardown == 1


def test_error_is_isolated_engine_survives_no_reload(tmp_path):
    _execute(tmp_path, "ref", "transformers", _spec(), _fake_get_cell)
    be = _execute(tmp_path, "cand", "vllm", _spec(), _fake_get_cell)
    assert be.n_load == 1                       # never reloaded to recover from the "a" error
    cells = score_runs(_spec(), str(tmp_path), "cand", "ref", quiet=True)
    by = {c.label: c for c in cells}
    assert by["a"].state == AppState.ERROR and "boom-a" in by["a"].error
    assert by["b"].state == AppState.SUPPORTED  # ran AFTER the error, same engine


def test_perf_meta_populated_for_ran_cells_only(tmp_path):
    _execute(tmp_path, "cand", "vllm", _spec(), _fake_get_cell)
    meta = load_run(str(tmp_path), "cand")[0][("__meta__",)]
    assert meta[("interactive", "b")]["error"] is None
    assert meta[("interactive", "b")]["median_latency_ms"] is not None
    assert meta[("interactive", "a")]["error"]                      # errored cell:
    assert "median_latency_ms" not in meta[("interactive", "a")]    # never timed


def test_batched_candidate_scored_against_perprompt_reference_not_padded_batch(tmp_path):
    """Batching is a coverage axis, and a single padded batch is NOT a valid per-prompt reference
    for absolute-position models: left-padding shifts the position embeddings on the padded rows.
    Every run therefore also writes the per-prompt stack (`batched_perprompt`), and score.py uses
    the REFERENCE's stack — not its padded batch — as the batched ground truth. Fixture: the hf
    padded batch corrupts prompt p2 (the position-shift effect) while the vllm batch runs each
    prompt as its own unpadded request. The vllm candidate matches the per-prompt truth ->
    SUPPORTED; scoring the hf run against itself compares its padded batch to its own per-prompt
    stack and flags the divergence."""
    per_prompt = {"p1": _onehot(1), "p2": _onehot(2), "p3": _onehot(3)}    # the per-prompt ground truth
    hf_batched = torch.cat([_onehot(1), _onehot(6), _onehot(3)], dim=0)    # hf padded batch: p2 corrupted
    vllm_batched = torch.cat([_onehot(1), _onehot(2), _onehot(3)], dim=0)  # vllm per-request: all correct

    def fake_get_cell(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            if len(prompts) == 1:                                  # per-prompt (reference stack)
                return per_prompt[prompts[0]].clone()
            return (vllm_batched if backend == "vllm_async" else hf_batched).clone()
        return fn

    spec = CellConfig(
        name="b", methodology="m", family="fam", repo="r",
        protocol_absence_reason="Synthetic batched fixture",
        regimes=[ExecutionRegime("batched", ["p1", "p2", "p3"])],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)

    _execute(tmp_path, "ref", "transformers", spec, fake_get_cell)
    _execute(tmp_path, "cand", "vllm", spec, fake_get_cell)
    cells = score_runs(spec, str(tmp_path), "cand", "ref", quiet=True)
    by = {c.workload: c for c in cells}
    assert by["batched"].state == AppState.SUPPORTED   # matches per-prompt truth, though != hf batch
    meta = load_run(str(tmp_path), "cand")[0][("__meta__",)]
    assert meta[("batched", "t")]["throughput"] is not None            # prompts/s reported

    # the reference's own padded batch diverges from its per-prompt stack; scoring the run against
    # itself (same engine -> the equivalence axis) surfaces exactly that padding divergence
    self_cells = score_runs(spec, str(tmp_path), "ref", "ref", quiet=True)
    assert {c.workload: c for c in self_cells}["batched"].state == AppState.DIVERGENT


def test_control_dtype_run_disambiguates_precision(tmp_path):
    """A low-precision candidate cell that diverges from the hf reference is SILENTLY_WRONG until
    a control-dtype run of the same engine disambiguates it: where the control-dtype output
    matches the hf reference, the divergence is precision (SUPPORTED_DEGRADED); where it also
    diverges, it is a real mechanism bug (stays SILENTLY_WRONG). Same logic as the old live fp32
    rerun, but from a run file (score.py --ctl)."""
    def cells_for(outputs):
        def get_cell(methodology, family, backend):
            def fn(impl, model, prompts, **params):
                return outputs[params.get("k")].clone()
            return fn
        return get_cell

    _execute(tmp_path, "ref", "transformers", _spec(), cells_for({"base": _onehot(0), "a": _onehot(3), "b": _onehot(2)}))
    # the candidate diverges from hf on BOTH tasks
    _execute(tmp_path, "cand", "vllm", _spec(), cells_for({"base": _onehot(0), "a": _onehot(5), "b": _onehot(6)}))
    # control-dtype run: matches hf for "a" (precision near-tie), diverges for "b" (real bug)
    _execute(tmp_path, "ctl", "vllm", _spec(), cells_for({"base": _onehot(0), "a": _onehot(3), "b": _onehot(6)}))

    cells = score_runs(_spec(), str(tmp_path), "cand", "ref", ctl="ctl", quiet=True)
    by = {c.label: c for c in cells}
    assert by["a"].state == AppState.SUPPORTED_DEGRADED   # ctl == hf -> precision
    assert by["b"].state == AppState.SILENTLY_WRONG       # ctl != hf -> real bug


def test_evaluate_equivalence_mode_emits_equivalent_divergent():
    """In equivalence mode (control != "hf") the oracle scores the candidate vs single-GPU vLLM and
    emits EQUIVALENT / DIVERGENT — never the correctness vocabulary (SUPPORTED / SILENTLY_WRONG). A
    candidate matching the control is EQUIVALENT even if both are 'wrong' vs HF; a candidate diverging
    is DIVERGENT (the real parallelism break a --pp/--tp run exists to catch)."""
    from isb.runner.run import CellResult, evaluate

    # task "a": candidate reproduces single-GPU; task "b": candidate diverges. evaluate scores per
    # (methodology, family, workload) group, so each task is its own evaluate call (as score.py does).
    ctrl_a = CellResult("m", "f", "vllm_async", "a", "RAN", value=_onehot(3), workload="interactive")
    cand_a = CellResult("m", "f", "vllm_pp", "a", "RAN", value=_onehot(3), workload="interactive")
    ctrl_b = CellResult("m", "f", "vllm_async", "b", "RAN", value=_onehot(3), workload="interactive")
    cand_b = CellResult("m", "f", "vllm_pp", "b", "RAN", value=_onehot(7), workload="interactive")
    evaluate([ctrl_a, cand_a], control="vllm_async")
    evaluate([ctrl_b, cand_b], control="vllm_async")
    assert ctrl_a.state == AppState.EQUIVALENT        # the single-GPU reference, trivially equivalent
    assert cand_a.state == AppState.EQUIVALENT        # candidate reproduces single-GPU
    assert cand_b.state == AppState.DIVERGENT         # candidate diverges from single-GPU
    assert cand_b.state not in (AppState.SUPPORTED, AppState.SILENTLY_WRONG)  # not the correctness axis


def test_evaluate_equivalence_within_noise_band():
    """In equivalence mode, a candidate whose softmax distributions MATCH within tolerance (tv <=
    tv_tol) but whose argmax flips on a few near-tie tokens (top1 < top1_thresh) is EQUIVALENT_DEGRADED
    — within TP reduction-order noise, not a real divergence. A genuine distributional divergence
    (tv > tv_tol) stays DIVERGENT, so the band does not swallow real breaks."""
    from isb.runner.run import CellResult, evaluate

    # 8 near-tie rows ([10.0, 9.98] -> softmax ~[.505,.495]); flipping a row's argmax barely moves the
    # distribution, so tv stays tiny while top1 drops.
    def near_tie(flip_rows):
        rows = []
        for i in range(8):
            a, b = (9.98, 10.0) if i in flip_rows else (10.0, 9.98)
            rows.append(torch.tensor([[a, b]]))
        return torch.cat(rows, dim=-2)  # [8, 2]

    ref = near_tie(set())
    ctrl = CellResult("m", "f", "vllm_async", "a", "RAN", value=ref.clone(), workload="interactive")
    noisy = CellResult("m", "f", "vllm_pp", "a", "RAN", value=near_tie({0, 1}), workload="interactive")
    evaluate([ctrl, noisy], control="vllm_async")
    assert noisy.state == AppState.EQUIVALENT_DEGRADED, (noisy.state, noisy.metrics)
    assert noisy.metrics["tv"] <= 0.05 and noisy.metrics["top1_agree"] < 0.9  # tv passes, top1 doesn't

    # a genuine divergence: distributions strongly disagree (tv large) -> DIVERGENT, NOT the band
    far_ref = torch.tensor([[12.0, 0.0]]).repeat(8, 1)
    far_cand = torch.tensor([[0.0, 12.0]]).repeat(8, 1)
    ctrl2 = CellResult("m", "f", "vllm_async", "b", "RAN", value=far_ref, workload="interactive")
    div = CellResult("m", "f", "vllm_pp", "b", "RAN", value=far_cand, workload="interactive")
    evaluate([ctrl2, div], control="vllm_async")
    assert div.state == AppState.DIVERGENT
    assert div.metrics["tv"] > 0.05


def test_aggregate_interactive_scores_over_all_prompts(tmp_path):
    """An aggregate-interactive workload runs each prompt as its OWN trace and scores the verdict over
    ALL of them — so a backend that is right on 1 prompt but wrong on another is caught, where a
    single-prompt (n=1) verdict would pass it. The vllm cell here matches hf on 3 of 4 prompts ->
    top-1 agreement 0.75 < 0.9 -> SILENTLY_WRONG (a single-prompt check on the first prompt would say
    SUPPORTED)."""
    refs = {"p0": _onehot(0), "p1": _onehot(1), "p2": _onehot(2), "p3": _onehot(3)}

    def fake(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            p = prompts[0]                                    # per-prompt: always a 1-element list
            if backend == "vllm_async" and p == "p3":
                return _onehot(7)                             # wrong on the 4th prompt only
            return refs[p].clone()
        return fn

    spec = CellConfig(
        name="agg", methodology="m", family="f", repo="r",
        protocol_absence_reason="Synthetic aggregate fixture",
        regimes=[ExecutionRegime("interactive", ["p0", "p1", "p2", "p3"])],   # aggregate=True (default)
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)

    _execute(tmp_path, "ref", "transformers", spec, fake)
    _execute(tmp_path, "cand", "vllm", spec, fake)
    cells = score_runs(spec, str(tmp_path), "cand", "ref", quiet=True)
    cell = {c.workload: c for c in cells}["interactive"]
    assert cell.state == AppState.SILENTLY_WRONG              # 3/4 agreement, caught
    assert cell.metrics["top1_agree"] == 0.75                 # aggregated over 4 prompts


def test_pair_unit_workload_aggregates_over_pairs(tmp_path):
    """A patching workload's unit is a (clean, corrupted) PAIR, not a single prompt. Execution runs
    each pair as its own trace (the cell consumes the 2-element pair) and aggregates the verdict over
    all pairs, so N pairs stack like N prompts. Here the vllm cell matches hf on 3 of 4 pairs -> top-1
    0.75 -> SILENTLY_WRONG, which a single-pair check would miss — the multi-trace point, for patching."""
    pairs = [("c0", "k0"), ("c1", "k1"), ("c2", "k2"), ("c3", "k3")]

    def fake(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            clean, corrupted = prompts                       # the cell receives the pair as a 2-list
            i = int(clean[1])                                # "c3" -> 3
            if backend == "vllm_async" and i == 3:
                return _onehot(7)                            # wrong on the 4th pair only
            return _onehot(i)
        return fn

    spec = CellConfig(
        name="patch_agg", methodology="activation_patching", family="f", repo="r",
        regimes=[ExecutionRegime("interactive", pairs, aggregate=True)],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)

    _execute(tmp_path, "ref", "transformers", spec, fake)
    _execute(tmp_path, "cand", "vllm", spec, fake)
    cells = score_runs(spec, str(tmp_path), "cand", "ref", quiet=True)
    cell = {c.workload: c for c in cells}["interactive"]
    assert cell.state == AppState.SILENTLY_WRONG
    assert cell.metrics["top1_agree"] == 0.75


def _run_all():
    print("uses pytest fixtures (tmp_path); run via pytest")


if __name__ == "__main__":
    _run_all()

"""Single-pass driver tests (isb/sweep/driver.py) — no GPU; fake backends + fake cells.

Verifies the harness invariants: ONE model load per backend amortized across tasks (not per cell);
an intervention error is isolated (the engine survives, later tasks still run, no reload); perf is
populated for the cells that ran and the oracle scores HF-vs-vLLM per (workload, task) on the warm
output.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.sweep.driver as driver  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, Workload  # noqa: E402
from isb.states import AppState  # noqa: E402

V = 8


def _onehot(i):
    t = torch.zeros(1, V); t[0, i] = 9.0; return t


# distinct outputs per task key; vLLM matches HF for "b" (SUPPORTED) and raises for "a" (ERROR)
OUT = {"base": _onehot(0), "a": _onehot(3), "b": _onehot(5)}


def _fake_get_cell(methodology, family, backend):
    def fn(impl, model, prompts, **params):
        k = params.get("k")
        if backend == "vllm_async" and k == "a":
            raise RuntimeError("boom-a")        # an isolated intervention error
        return OUT[k].clone()
    return fn


class _FakeBackend:
    def __init__(self, name):
        self.name = name
        self.n_load = 0
        self.n_teardown = 0

    def load(self, repo):
        self.n_load += 1
        return f"model::{self.name}"

    def teardown(self, model):
        self.n_teardown += 1


def _spec():
    return CellConfig(
        name="fake", methodology="m", family="fam", repo="repo://x",
        workloads=[Workload("interactive", ["one prompt"])],
        tasks=[({"k": "a"}, "a"), ({"k": "b"}, "b")],   # task "a" errors on vLLM, "b" succeeds AFTER it
        baseline=BaselineSpec(params={"k": "base"}),
        effect=None, warmup=0, n_trials=1,
    )


def _run():
    created = {}

    def factory(name, spec):
        b = _FakeBackend(name); created[name] = b; return b

    orig = driver.get_cell
    driver.get_cell = _fake_get_cell
    try:
        results = driver.run_sweep(_spec(), backends=("hf", "vllm_async"), backend_factory=factory)
    finally:
        driver.get_cell = orig
    return results, created


def test_model_loaded_once_per_backend_not_per_cell():
    _, created = _run()
    assert created["hf"].n_load == 1            # amortized across both tasks, not 2 loads
    assert created["vllm_async"].n_load == 1
    assert created["hf"].n_teardown == 1
    assert created["vllm_async"].n_teardown == 1


def test_error_is_isolated_engine_survives_no_reload():
    results, created = _run()
    by = {(c.backend, c.label): c for c in results}
    assert by[("vllm_async", "a")].state == AppState.ERROR     # the failing task
    assert by[("vllm_async", "b")].state == AppState.SUPPORTED  # ran AFTER the error, same engine
    assert created["vllm_async"].n_load == 1                    # never reloaded to recover


def test_perf_populated_for_ran_cells_only():
    results, _ = _run()
    by = {(c.backend, c.label): c for c in results}
    assert by[("hf", "a")].perf is not None and by[("hf", "a")].perf.n_trials == 1
    assert by[("vllm_async", "b")].perf is not None
    assert by[("vllm_async", "a")].perf is None                 # ERROR cell never timed


def test_oracle_scores_per_task_on_warm_output():
    results, _ = _run()
    by = {(c.backend, c.label): c for c in results}
    assert by[("hf", "a")].state == AppState.SUPPORTED          # control
    assert by[("hf", "b")].state == AppState.SUPPORTED
    assert by[("vllm_async", "b")].state == AppState.SUPPORTED  # vLLM "b" == HF "b" -> equivalent


def test_batched_oracle_scores_against_per_prompt_reference_not_padded_batch():
    """Batching is a coverage axis, and a single padded batch is NOT a valid per-prompt reference
    for absolute-position models: left-padding shifts the position embeddings on the padded rows, so
    a backend's own batched output can disagree with its per-prompt-interactive result. The oracle
    must therefore score EVERY batched cell — the HF control included — against the per-prompt
    reference (the cell run on each prompt alone), not auto-pass HF-batched.

    Fixture: each cell returns a per-prompt row for a 1-element prompt list (the reference is the
    concat of those), and a *different* output for the full batch. HF-batched diverges from its own
    per-prompt truth on one prompt (the position-shift effect) and must be flagged; vLLM runs each
    prompt as its own unpadded request, so its batched output matches the per-prompt truth and is
    SUPPORTED. The old oracle, which trusted HF-batched as the control, could catch neither."""
    per_prompt = {"p1": _onehot(1), "p2": _onehot(2), "p3": _onehot(3)}    # the per-prompt ground truth
    hf_batched = torch.cat([_onehot(1), _onehot(6), _onehot(3)], dim=0)    # HF padded batch: p2 corrupted
    vllm_batched = torch.cat([_onehot(1), _onehot(2), _onehot(3)], dim=0)  # vLLM per-request: all correct

    def fake_get_cell(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            if len(prompts) == 1:                                  # per-prompt (reference + interactive)
                return per_prompt[prompts[0]].clone()
            return (vllm_batched if backend == "vllm_async" else hf_batched).clone()   # padded batch
        return fn

    spec = CellConfig(
        name="b", methodology="m", family="fam", repo="r",
        workloads=[Workload("batched", ["p1", "p2", "p3"])],
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)

    created = {}
    orig_gc, orig_rc = driver.get_cell, driver.run_cell
    driver.get_cell = fake_get_cell
    driver.run_cell = lambda *a, **k: type("R", (), {"value": None})()    # fp32 rerun -> no reclassify, no GPU
    try:
        results = driver.run_sweep(
            spec, backends=("hf", "vllm_async"),
            backend_factory=lambda name, s: created.setdefault(name, _FakeBackend(name)))
    finally:
        driver.get_cell, driver.run_cell = orig_gc, orig_rc

    by = {(c.backend, c.workload): c for c in results}
    # HF-batched is NOT auto-passed: its padded batch diverges from the per-prompt truth -> flagged.
    assert by[("hf", "batched")].state == AppState.SILENTLY_WRONG
    # vLLM-batched matches the per-prompt truth -> SUPPORTED even though it differs from HF-batched.
    assert by[("vllm_async", "batched")].state == AppState.SUPPORTED
    assert by[("vllm_async", "batched")].perf.throughput is not None        # prompts/s reported
    assert by[("hf", "batched")].perf.throughput is not None


def test_vm_serve_scores_against_cached_integrated_refs():
    """VM mode: the GPU-less serve client has no in-run HF control, so it scores serve cells against
    references cached by a prior integrated run. Round-trip: (1) an integrated run dumps the HF
    interactive control output per task; (2) a serve-only run loads those refs and scores its cells
    against them — SUPPORTED when the serve output matches the cached HF reference, SILENTLY_WRONG when
    it diverges. Uses fakes (no GPU, no server)."""
    import tempfile

    # the serve backend's fake cell: matches the cached HF ref for task "b", diverges for task "a"
    def fake_get_cell(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            k = params.get("k")
            if backend == "vllm_serve":
                return (OUT["b"].clone() if k == "b" else _onehot(7))  # "a" -> wrong vs HF ref OUT["a"]
            return OUT[k].clone()                                       # hf / vllm_async truth
        return fn

    with tempfile.TemporaryDirectory() as d:
        orig = driver.get_cell
        driver.get_cell = fake_get_cell
        try:
            # 1) integrated run dumps the HF reference for each task
            driver.run_sweep(
                _spec(), backends=("hf",), dump_refs=d,
                backend_factory=lambda name, s: _FakeBackend(name))
            # 2) serve-only run scores against the cached refs (no HF control present this run)
            results = driver.run_sweep(
                _spec(), backends=("vllm_serve",), serve_host="http://x:6677", refs=d,
                backend_factory=lambda name, s: _FakeBackend(name))
        finally:
            driver.get_cell = orig

    by = {(c.backend, c.label): c for c in results}
    assert by[("vllm_serve", "b")].state == AppState.SUPPORTED        # matches cached HF reference
    assert by[("vllm_serve", "a")].state == AppState.SILENTLY_WRONG   # diverges from cached HF reference


def test_vm_serve_precision_disambiguation_via_ctl_refs():
    """A bf16 serve cell that diverges from the HF control is SILENTLY_WRONG, but the GPU-less client
    disambiguates with the cached fp32-vLLM (control-dtype) output: if THAT matches the HF control the
    divergence is a precision near-tie -> SUPPORTED_DEGRADED; if the fp32-vLLM output also diverges it's
    a real mechanism bug -> stays SILENTLY_WRONG. (The GPU-less analog of disambiguate_precision's live
    fp32 rerun — this is what separates ablation/patching bf16 from the llama dual-residual bug.)"""
    import tempfile

    import torch

    def fake_get_cell(methodology, family, backend):
        def fn(impl, model, prompts, **params):
            k = params.get("k")
            return _onehot(5) if k == "a" else _onehot(6)   # both diverge from HF -> both SILENTLY_WRONG
        return fn

    with tempfile.TemporaryDirectory() as d:
        # HF control (fp32 truth)
        torch.save({("interactive", "a"): _onehot(3), ("interactive", "b"): _onehot(2)},
                   driver._ref_file(d, "fake"))
        # fp32-vLLM control-dtype output: matches HF for "a" (precision), diverges for "b" (real bug)
        torch.save({("interactive", "a"): _onehot(3), ("interactive", "b"): _onehot(6)},
                   driver._ctl_ref_file(d, "fake"))
        orig = driver.get_cell
        driver.get_cell = fake_get_cell
        try:
            results = driver.run_sweep(
                _spec(), backends=("vllm_serve",), serve_host="http://x:6677",
                refs=d, ctl_refs=d, backend_factory=lambda name, s: _FakeBackend(name))
        finally:
            driver.get_cell = orig

    by = {(c.backend, c.label): c for c in results}
    assert by[("vllm_serve", "a")].state == AppState.SUPPORTED_DEGRADED  # fp32-vLLM == HF -> precision
    assert by[("vllm_serve", "b")].state == AppState.SILENTLY_WRONG      # fp32-vLLM != HF -> real bug


def test_expected_state_reports_only_the_delta():
    """A run's headline is the DELTA vs declared expectation: a cell that matches its expected state is
    NOT a surprise; one that diverges IS. (This is the structural antidote to restating documented
    gaps as findings.)"""
    spec = CellConfig(
        name="exp", methodology="m", family="fam", repo="r",
        workloads=[Workload("interactive", ["p"])],
        tasks=[({"k": "a"}, "a"), ({"k": "b"}, "b")],
        baseline=BaselineSpec(params={"k": "base"}), effect=None, warmup=0, n_trials=1,
        expected={("vllm_async", "interactive", "a"): "ERROR"})   # "a" is a declared frontier

    created = {}
    orig = driver.get_cell
    driver.get_cell = _fake_get_cell                              # vllm_async "a" raises -> ERROR
    try:
        results = driver.run_sweep(spec, backends=("hf", "vllm_async"),
                                   backend_factory=lambda n, s: created.setdefault(n, _FakeBackend(n)))
    finally:
        driver.get_cell = orig

    by = {(c.backend, c.label): c for c in results}
    assert by[("vllm_async", "a")].state == AppState.ERROR
    assert by[("vllm_async", "a")].surprise is False             # ERROR == declared ERROR -> no news
    assert by[("vllm_async", "b")].surprise is False             # SUPPORTED == default -> no news
    # drop the declaration: the same ERROR now IS a surprise
    spec.expected = {}
    driver.get_cell = _fake_get_cell
    try:
        results2 = driver.run_sweep(spec, backends=("hf", "vllm_async"),
                                    backend_factory=lambda n, s: _FakeBackend(n))
    finally:
        driver.get_cell = orig
    assert {(c.backend, c.label): c.surprise for c in results2}[("vllm_async", "a")] is True


def test_expected_state_serve_inherits_vllm_async():
    """A vllm_serve cell with no own expectation inherits the vllm_async one — serve should match
    in-process vLLM, so a divergence over the wire is a genuine transport surprise."""
    from isb.sweep.driver import expected_state

    spec = CellConfig(
        name="x", methodology="m", family="f", repo="r",
        workloads=[Workload("interactive", ["p"])], tasks=[({}, "a")],
        baseline=BaselineSpec(params={}), effect=None,
        expected={("vllm_async", "interactive", "a"): "ERROR"})
    assert expected_state(spec, "vllm_serve", "interactive", "a") == "ERROR"     # inherits async
    assert expected_state(spec, "vllm_serve", "interactive", "z") == "SUPPORTED"  # unlisted -> default
    assert expected_state(spec, "hf", "interactive", "a") == "SUPPORTED"          # hf != async


def test_expected_state_separates_equivalence_from_correctness():
    """A --pp/--tp run measures the PARALLELISM-EQUIVALENCE axis, not correctness, so expected_state
    must NOT collapse a vs-HF correctness state into SUPPORTED — that overload is exactly what stamped a
    known-wrong read 'SUPPORTED'. In equivalence mode every cell that RUNS is expected EQUIVALENT (the
    candidate should reproduce single-GPU, naive-vs-HF read and all); only ERROR carries over (a cell
    errors on both topologies). The declared vs-HF correctness stays available via `_declared`,
    untouched, for display. Correctness mode (control="hf") returns the declared state verbatim."""
    from isb.sweep.driver import _declared, expected_state

    spec = CellConfig(
        name="p", methodology="m", family="f", repo="r",
        workloads=[Workload("interactive", ["p"])],
        tasks=[({}, "wrong"), ({}, "degraded"), ({}, "err")],
        baseline=BaselineSpec(params={}), effect=None,
        expected={
            ("vllm_async", "interactive", "wrong"): AppState.SILENTLY_WRONG,
            ("vllm_async", "interactive", "degraded"): AppState.SUPPORTED_DEGRADED,
            ("vllm_async", "interactive", "err"): AppState.ERROR,
        })

    # equivalence mode (control != "hf"): a cell that RUNS -> EQUIVALENT (never SUPPORTED), however
    # wrong it is vs HF; a cell that ERRORs on both topologies -> ERROR.
    assert expected_state(spec, "vllm_pp", "interactive", "wrong", control="vllm_async") == AppState.EQUIVALENT
    assert expected_state(spec, "vllm_pp", "interactive", "degraded", control="vllm_async") == AppState.EQUIVALENT
    assert expected_state(spec, "vllm_pp", "interactive", "err", control="vllm_async") == AppState.ERROR
    assert expected_state(spec, "vllm_pp", "interactive", "wrong", control="vllm_async") != AppState.SUPPORTED
    # the declared vs-HF correctness is preserved (display), NOT coerced
    assert _declared(spec, "vllm_pp", "interactive", "wrong") == AppState.SILENTLY_WRONG       # inherits async
    assert _declared(spec, "vllm_pp", "interactive", "degraded") == AppState.SUPPORTED_DEGRADED
    assert _declared(spec, "vllm_pp", "interactive", "z") == AppState.SUPPORTED                # unlisted -> default
    # correctness mode (control="hf") is the declared state verbatim
    assert expected_state(spec, "vllm_async", "interactive", "wrong", control="hf") == AppState.SILENTLY_WRONG
    assert expected_state(spec, "vllm_async", "interactive", "degraded", control="hf") == AppState.SUPPORTED_DEGRADED


def test_evaluate_equivalence_mode_emits_equivalent_divergent():
    """In equivalence mode (control != "hf") the oracle scores the candidate vs single-GPU vLLM and
    emits EQUIVALENT / DIVERGENT — never the correctness vocabulary (SUPPORTED / SILENTLY_WRONG). A
    candidate matching the control is EQUIVALENT even if both are 'wrong' vs HF; a candidate diverging
    is DIVERGENT (the real parallelism break a --pp/--tp run exists to catch)."""
    from isb.runner.run import CellResult, evaluate

    # task "a": candidate reproduces single-GPU; task "b": candidate diverges. evaluate scores per
    # (methodology, family, workload) group, so each task is its own evaluate call (as the driver does).
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
    import torch

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


def test_aggregate_interactive_scores_over_all_prompts():
    """An aggregate-interactive workload runs each prompt as its OWN trace and scores the verdict over
    ALL of them — so a backend that is right on 1 prompt but wrong on another is caught, where a
    single-prompt (n=1) verdict would pass it. vLLM here matches HF on 3 of 4 prompts -> top-1
    agreement 0.75 < 0.9 -> SILENTLY_WRONG (a single-prompt check on the first prompt would say
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
        workloads=[Workload("interactive", ["p0", "p1", "p2", "p3"])],   # aggregate=True (default)
        tasks=[({}, "t")], baseline=BaselineSpec(params={}), effect=None, warmup=0, n_trials=1)

    orig_gc, orig_fp = driver.get_cell, driver._fp32_rerun
    driver.get_cell = fake
    driver._fp32_rerun = lambda *a, **k: (lambda c: None)   # disambiguation rerun -> None: no GPU, no reclassify
    try:
        results = driver.run_sweep(spec, backends=("hf", "vllm_async"),
                                   backend_factory=lambda n, s: _FakeBackend(n))
    finally:
        driver.get_cell, driver._fp32_rerun = orig_gc, orig_fp

    by = {(c.backend, c.workload): c for c in results}
    assert by[("hf", "interactive")].state == AppState.SUPPORTED            # control
    assert by[("vllm_async", "interactive")].state == AppState.SILENTLY_WRONG  # 3/4 agreement, caught
    assert by[("vllm_async", "interactive")].metrics["top1_agree"] == 0.75    # aggregated over 4 prompts


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

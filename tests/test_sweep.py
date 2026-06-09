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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

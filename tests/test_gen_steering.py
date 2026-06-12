"""Generation-time steering tests (isb/methodologies/gen_steering.py + the generation workload
plumbing) — no GPU; torch + fakes only.

Pins (1) the per-step replacement-write semantics (the part whose vLLM divergence the GPU sweep
measures), (2) the generation Workload validation + the driver's new_tokens injection (the regime
axis lives on the Workload, cells must still receive it), and (3) a fake-backend sweep over a
generation workload: oracle on [steps, vocab] stacks, tokens/s throughput, effect-size in the
generation regime. The bounded-vs-unbounded iteration REALIZATION itself is backend behavior —
exercised by the GPU run, not fakeable here.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
import isb.sweep.driver as driver  # noqa: E402
from isb.methodologies.gen_steering import _check_bound, _steer_step  # noqa: E402
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402
from isb.states import AppState  # noqa: E402
from isb.sweep.driver import _task_params, _throughput  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload  # noqa: E402

VOCAB, HID = 11, 4


class _Block:
    def __init__(self, out):
        self.output = out


class _Head:
    def __init__(self):
        self.weight = torch.eye(VOCAB, HID)  # row i is the unit direction e_i


def test_cells_registered_and_serve_falls_back():
    assert ("gen_steering", "gpt2", "hf") in CELLS
    assert ("gen_steering", "gpt2", "vllm_async") in CELLS
    # the serve backend reuses the in-process vLLM cell by construction
    assert get_cell("gen_steering", "gpt2", "vllm_serve") is CELLS[
        ("gen_steering", "gpt2", "vllm_async")]


def test_generation_workload_validates():
    w = Workload("generation", ["p"], new_tokens=8)
    assert w.aggregate                                   # per-prompt traces, stacked verdict
    try:
        Workload("generation", ["p"])                    # new_tokens defaults to 0
        raise AssertionError("generation with new_tokens=0 must raise")
    except ValueError:
        pass
    try:
        Workload("streaming", ["p"])
        raise AssertionError("unknown kind must raise")
    except ValueError:
        pass


def test_driver_injects_new_tokens_from_workload():
    gen = Workload("generation", ["p"], new_tokens=5)
    assert _task_params(gen, {"alpha": 1.0}) == {"alpha": 1.0, "new_tokens": 5}
    inter = Workload("interactive", ["p"])
    assert _task_params(inter, {"alpha": 1.0}) == {"alpha": 1.0}   # untouched off-generation


def test_generation_throughput_is_tokens_per_second():
    class _T:
        median_ms = 500.0
    assert _throughput(Workload("generation", ["p"], new_tokens=8), _T()) == 16.0  # 8 tok / 0.5 s
    assert _throughput(Workload("interactive", ["p"]), _T()) is None


def test_steer_step_replacement_write_semantics():
    hidden = torch.ones(2, HID)
    before = hidden.clone()
    blocks = [_Block((hidden, "kv"))]
    _steer_step(blocks, _Head(), layer=0, token_id=1, alpha=3.0)
    new_hidden, tail = blocks[0].output[0], blocks[0].output[1]
    assert tail == "kv"                                   # tuple tail preserved
    assert new_hidden.data_ptr() != hidden.data_ptr()     # replacement: NEW tensor,
    assert torch.equal(hidden, before)                    # the live buffer untouched
    # direction = e_1; scale = mean per-token norm of ones(2,4) = 2.0 -> add 3*2 on dim 1
    expect = before + torch.tensor([0.0, 6.0, 0.0, 0.0])
    assert torch.allclose(new_hidden, expect)


def test_steer_step_handles_plain_tensor_output():
    hidden = torch.ones(2, HID)
    blocks = [_Block(hidden)]
    _steer_step(blocks, _Head(), layer=0, token_id=2, alpha=1.0)
    assert not isinstance(blocks[0].output, tuple)
    assert torch.allclose(blocks[0].output, hidden + torch.tensor([0.0, 0.0, 2.0, 0.0]))


def test_unknown_bound_rejected():
    try:
        _check_bound("forever")
        raise AssertionError("unknown bound must raise")
    except ValueError:
        pass


class _GenModel:
    """Just enough model for the HF cell: tokenizer, block list, lm_head with weight + a per-step
    refreshed .output (the fake backend below sets it before each step)."""

    class _TF:
        def __init__(self, h):
            self.h = h

    def __init__(self, n_blocks=2):
        self.transformer = self._TF([_Block(torch.ones(2, HID)) for _ in range(n_blocks)])
        self.lm_head = _Head()
        self.lm_head.output = torch.zeros(1, 3, VOCAB)

    def tokenizer(self, text, add_special_tokens=False):
        return {"input_ids": [1]}


class _GenBackend:
    """Fake be.generate: records the plumbing args and drives build_step like the real loop."""

    def __init__(self):
        self.calls = []

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        self.calls.append({"prompts": prompts, "new_tokens": new_tokens, "bounded": bounded})
        rows = [build_step() for _ in range(new_tokens)]
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)


def test_hf_cell_plumbs_bound_and_new_tokens_and_steers():
    be, model = _GenBackend(), _GenModel()
    fn = get_cell("gen_steering", "gpt2", "hf")
    out = fn(be, model, ["p"], layer=0, target="x", alpha=2.0, bound="unbounded", new_tokens=4)
    assert be.calls[0]["new_tokens"] == 4
    assert be.calls[0]["bounded"] is False                # bound="unbounded" -> bounded=False
    assert out.shape == (4, VOCAB)                        # per-step rows stacked
    assert isinstance(model.transformer.h[0].output, tuple) or not torch.equal(
        model.transformer.h[0].output, torch.ones(2, HID))   # the write landed


def test_hf_cell_alpha_zero_is_pure_readout():
    be, model = _GenBackend(), _GenModel()
    before = model.transformer.h[0].output.clone()
    fn = get_cell("gen_steering", "gpt2", "hf")
    out = fn(be, model, ["p"], layer=0, target="x", alpha=0.0, bound="bounded", new_tokens=3)
    assert torch.equal(model.transformer.h[0].output, before)   # no write at all
    assert out.shape == (3, VOCAB)


# ---- fake-backend sweep over a generation workload -------------------------------------------

V = 8


def _logits(seed, steps=4):
    t = torch.zeros(steps, V)
    t[:, seed] = 9.0
    return t


def _fake_get_cell(methodology, family, backend):
    def fn(impl, model, prompts, **params):
        assert params["new_tokens"] == 4, "driver must inject the workload's new_tokens"
        if params.get("alpha") == 0.0:
            return _logits(0)                       # unsteered baseline distribution
        return _logits(3 if params["bound"] == "bounded" else 5)
    return fn


class _FakeBackend:
    def __init__(self, name):
        self.name = name

    def load(self, repo):
        return f"model::{self.name}"

    def teardown(self, model):
        pass


def test_generation_sweep_oracle_throughput_and_effect():
    spec = CellConfig(
        name="fake_gen", methodology="m", family="fam", repo="repo://x",
        workloads=[Workload("generation", ["p1", "p2"], new_tokens=4)],
        tasks=[({"alpha": 6.0, "bound": "bounded"}, "bound=iter[0:N]"),
               ({"alpha": 6.0, "bound": "unbounded"}, "bound=iter[:]")],
        baseline=BaselineSpec(params={"alpha": 0.0, "bound": "bounded"}),
        effect=EffectSpec(baseline_params={"alpha": 0.0, "bound": "bounded"},
                          perturbed_params={"alpha": 6.0, "bound": "bounded"}),
        warmup=0, n_trials=1,
    )
    orig_gc, orig_fp = driver.get_cell, driver._fp32_rerun
    driver.get_cell = _fake_get_cell
    driver._fp32_rerun = lambda *a, **k: (lambda c: None)
    try:
        results = driver.run_sweep(spec, backends=("hf", "vllm_async"),
                                   backend_factory=lambda n, s: _FakeBackend(n))
    finally:
        driver.get_cell, driver._fp32_rerun = orig_gc, orig_fp

    by = {(c.backend, c.label): c for c in results}
    for be_name in ("hf", "vllm_async"):                  # same fake output both backends
        assert by[(be_name, "bound=iter[0:N]")].state == AppState.SUPPORTED
        assert by[(be_name, "bound=iter[:]")].state == AppState.SUPPORTED
    p = by[("vllm_async", "bound=iter[0:N]")].perf
    assert p is not None and p.throughput is not None     # tokens/s populated for generation
    assert by[("hf", "bound=iter[0:N]")].workload == "generation"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

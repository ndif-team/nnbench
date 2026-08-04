"""Generation-time cross-prompt patching tests (isb/methodologies/gen_patching.py + the
generate_patch backend plumbing) — no GPU; torch + fakes only.

Pins (1) the prefill-only injection semantics (the shape-gated write that lands at prefill and
leaves decode steps untouched — the part the GPU run then measures across backends), (2) the
clean-snapshot capture, (3) the cell plumbing through a fake generate_patch (capture once, inject
per step, baseline path uses plain generate with no capture), and (4) a fake-backend sweep over a
generation workload consuming a clean/corrupt PAIR (aggregate=False). The composition's actual
cross-backend verdict is the GPU run's job, not fakeable here.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
import isb.methodologies.gen_patching as gen_patching  # noqa: E402
import isb.sweep.execute as execute_mod  # noqa: E402
from isb.methodologies.gen_patching import (  # noqa: E402
    _capture,
    _check_bound,
    _inject_transplant,
)
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402
from isb.states import AppState  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload  # noqa: E402

HID, VOCAB = 4, 8


class _Block:
    def __init__(self, out):
        self.output = out


def test_cells_registered_and_serve_falls_back():
    # family-generic (§12.8): registered once under family="*", resolved per profiled family
    assert ("gen_patching", "*", "hf") in CELLS
    assert ("gen_patching", "*", "vllm_async") in CELLS
    # serve reuses the in-process vLLM cell: same underlying generic fn behind the profile-bound wrapper
    assert get_cell("gen_patching", "gpt2", "vllm_serve").__wrapped__ is CELLS[
        ("gen_patching", "*", "vllm_async")]


def test_inject_replaces_residual():
    clean = torch.arange(3 * HID, dtype=torch.float32).reshape(3, HID)   # a 3-token prompt residual
    blocks = [_Block((torch.zeros(3, HID), "kv"))]
    _inject_transplant(blocks, 0, clean)
    new_hidden, tail = blocks[0].output[0], blocks[0].output[1]
    assert tail == "kv"                                  # tuple tail preserved
    assert torch.equal(new_hidden, clean)                # the transplant landed (whole-tuple replace)


def test_inject_raises_on_length_mismatch():
    clean = torch.arange(3 * HID, dtype=torch.float32).reshape(3, HID)   # captured 3-token residual
    blocks = [_Block((torch.zeros(1, HID), "kv"))]       # a 1-token residual -> shapes disagree
    try:
        _inject_transplant(blocks, 0, clean)
        raise AssertionError("mismatched clean/corrupted lengths must raise, not silently no-op")
    except ValueError:
        pass


def test_inject_handles_plain_tensor_output():
    clean = torch.ones(2, HID)
    blocks = [_Block(torch.zeros(2, HID))]               # non-tuple block output
    _inject_transplant(blocks, 0, clean)
    assert not isinstance(blocks[0].output, tuple)
    assert torch.equal(blocks[0].output, clean)


def test_capture_is_an_independent_snapshot():
    resid = torch.ones(2, HID)
    blocks = [_Block((resid, "kv"))]
    snap = _capture(blocks, 0, "plain")
    resid.add_(5.0)                                      # mutate the live buffer after capture
    assert torch.equal(snap, torch.ones(2, HID))         # snapshot unaffected -> it cloned


def test_unknown_bound_rejected():
    try:
        _check_bound("forever")
        raise AssertionError("unknown bound must raise")
    except ValueError:
        pass


# ---- cell plumbing through a fake generate_patch ----------------------------------------------

class _Head:
    pass


class _TF:
    def __init__(self, h):
        self.h = h


class _PatchModel:
    """A 2-block model whose layer-0 output is a tuple matching the captured prompt residual, so the
    cell's transplant actually fires at the (fake) prefill; lm_head/logits are fixed reads."""

    def __init__(self, prompt_len=3):
        self.transformer = _TF([_Block((torch.zeros(prompt_len, HID), "kv")) for _ in range(2)])
        self.lm_head = _Head()
        self.lm_head.output = torch.zeros(1, 4, VOCAB)
        self.logits = torch.zeros(4, VOCAB)


class _PatchBackend:
    """Fake generate_patch / generate: record args and drive the closures like the real loops."""

    def __init__(self):
        self.calls = []

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        self.calls.append({"kind": "generate", "prompts": prompts,
                           "new_tokens": new_tokens, "bounded": bounded})
        rows = [build_step() for _ in range(new_tokens)]
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)

    def generate_patch(self, model, source_prompt, base_prompt, capture, build_step,
                       *, new_tokens, bounded=True):
        self.calls.append({"kind": "generate_patch", "source": source_prompt,
                           "base": base_prompt, "new_tokens": new_tokens, "bounded": bounded})
        clean = capture()                                # one capture, like trace 1
        rows = [build_step(clean) for _ in range(new_tokens)]
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)


def test_patched_cell_captures_then_injects():
    be, model = _PatchBackend(), _PatchModel(prompt_len=3)
    fn = get_cell("gen_patching", "gpt2", "hf")
    out = fn(be, model, ["clean prompt", "corrupt prompt"],
             layer=0, bound="unbounded", new_tokens=4, patch=True)
    c = be.calls[0]
    assert c["kind"] == "generate_patch"
    assert c["source"] == "clean prompt" and c["base"] == "corrupt prompt"
    assert c["new_tokens"] == 4 and c["bounded"] is False       # bound="unbounded" -> bounded=False
    assert out.shape == (4, VOCAB)
    # the capture (layer-0 residual) was injected at the fake prefill (whole-tuple replace)
    assert torch.equal(model.transformer.h[0].output[0], torch.zeros(3, HID))


def test_cell_injects_exactly_once_even_for_one_token_prompt():
    # Regression: shape-only prefill detection re-injects on EVERY decode step of a one-token prompt
    # (decode hidden shares the prefill's [1, hidden] shape). The first-forward flag must inject
    # exactly once regardless of prompt length — else the method silently changes outside the
    # length>1 benchmark prompts.
    be, model = _PatchBackend(), _PatchModel(prompt_len=1)
    calls = []
    orig = gen_patching._inject_transplant
    gen_patching._inject_transplant = lambda *a, **k: calls.append(1)
    try:
        fn = get_cell("gen_patching", "gpt2", "hf")
        fn(be, model, ["a", "b"], layer=0, bound="bounded", new_tokens=5, patch=True)
    finally:
        gen_patching._inject_transplant = orig
    assert calls == [1], f"expected exactly one injection over 5 decode steps, got {len(calls)}"


def test_baseline_cell_uses_generate_with_no_capture():
    be, model = _PatchBackend(), _PatchModel()
    fn = get_cell("gen_patching", "gpt2", "hf")
    out = fn(be, model, ["clean", "corrupt"], layer=0, bound="bounded", new_tokens=3, patch=False)
    assert be.calls[0]["kind"] == "generate"                    # baseline path, no transplant
    assert be.calls[0]["prompts"] == ["corrupt"]                # the corrupted run alone
    assert out.shape == (3, VOCAB)


def test_vllm_cell_reads_engine_logits_site():
    be, model = _PatchBackend(), _PatchModel(prompt_len=3)
    fn = get_cell("gen_patching", "gpt2", "vllm_async")
    out = fn(be, model, ["clean", "corrupt"], layer=0, bound="bounded", new_tokens=2, patch=True)
    assert be.calls[0]["kind"] == "generate_patch"
    assert out.shape == (2, VOCAB)                               # reads model.logits[-1:, :]


# ---- fake-backend sweep over a generation workload (clean/corrupt pair) ------------------------

def _logits(seed, steps=5):
    t = torch.zeros(steps, VOCAB)
    t[:, seed] = 9.0
    return t


def _fake_get_cell(methodology, family, backend):
    def fn(impl, model, prompts, **params):
        assert params["new_tokens"] == 5, "the execute layer must inject the workload's new_tokens"
        assert isinstance(prompts, (list, tuple)) and len(prompts) == 2, "pair consumed as one unit"
        if not params.get("patch", True):
            return _logits(0)                            # unpatched-generation baseline
        return _logits(3 if params["bound"] == "bounded" else 6)
    return fn


class _FakeBackend:
    def load(self, repo):
        return "model"

    def teardown(self, model):
        pass


def test_generation_pair_sweep_oracle_and_effect():
    import tempfile

    from isb.runfile import load_run
    from isb.runs import EngineConfig, RunConfig
    from isb.sweep.execute import execute_run
    from isb.sweep.score import score_runs

    spec = CellConfig(
        name="fake_gp", methodology="m", family="fam", repo="repo://x",
        workloads=[Workload("generation", ["CLEAN", "CORRUPT"], new_tokens=5, aggregate=False)],
        tasks=[({"bound": "bounded", "patch": True}, "bound=iter[0:N]"),
               ({"bound": "unbounded", "patch": True}, "bound=iter[:]")],
        baseline=BaselineSpec(params={"bound": "bounded", "patch": False}),
        effect=EffectSpec(baseline_params={"bound": "bounded", "patch": False},
                          perturbed_params={"bound": "bounded", "patch": True}),
        warmup=0, n_trials=1,
    )

    def execute_fake(out_dir, name, engine_kind):
        orig = execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance
        execute_mod.get_cell = _fake_get_cell
        execute_mod.make_backend = lambda run, sp: _FakeBackend()
        execute_mod.resolve_provenance = lambda run: {
            "client": {"nnsight": {"commit": "c"}, "vllm": None, "transformers": "5.x"},
            "deployment": {"kind": "local"},
            "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": {}},
            "host": {"hostname": "testbox", "gpus": []},
        }
        try:
            execute_run(spec, RunConfig(engine=EngineConfig(engine_kind)), out_dir, name)
        finally:
            execute_mod.get_cell, execute_mod.make_backend, execute_mod.resolve_provenance = orig

    with tempfile.TemporaryDirectory() as d:
        execute_fake(d, "hf", "transformers")
        execute_fake(d, "vllm", "vllm")
        cells = score_runs(spec, d, "vllm", "hf", quiet=True)
        meta = load_run(d, "vllm")[0][("__meta__",)]

    by = {c.label: c for c in cells}
    assert by["bound=iter[0:N]"].state == AppState.SUPPORTED   # same fake output both engines
    assert by["bound=iter[0:N]"].workload == "generation"
    m = meta[("generation", "bound=iter[0:N]")]
    assert m["throughput"] is not None                         # tokens/s populated for generation
    assert meta[("__effect__", "generation")]["strong"]        # the patch moves the fake control


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

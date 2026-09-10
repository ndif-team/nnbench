"""Case descriptors agree with explicit cell branches, using CPU tensors and backend spies."""
import inspect
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.methodologies.registry import get_cell  # noqa: E402
from isb.methodologies.requirements import (  # noqa: E402
    CASE_DESCRIPTIONS, case_description, describe_case,
)
from isb.protocol import PROTOCOLS, InterventionSpec  # noqa: E402


class OutputSpy:
    def __init__(self, output):
        self._output = output
        self.writes = 0

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self.writes += 1
        self._output = value


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}

    def encode(self, text):
        return [1]


def model_fixture():
    blocks = [OutputSpy(torch.ones(2, 3)) for _ in range(2)]
    for block in blocks:
        block.mlp = OutputSpy(torch.ones(2, 3))
        block.attn = OutputSpy(torch.ones(2, 3))
    return SimpleNamespace(
        transformer=SimpleNamespace(h=blocks, ln_f=lambda x: x),
        lm_head=SimpleNamespace(weight=torch.eye(3), output=torch.ones(1, 2, 3)),
        logits=torch.ones(2, 3), tokenizer=Tokenizer(), config=SimpleNamespace(n_embd=3),
    )


class GradientObserved(Exception):
    pass


class BackendSpy:
    name = "hf"

    def __init__(self):
        self.calls = []

    def last(self, value):
        return value[-1:, :]

    def run(self, model, prompts, build):
        self.calls.append("run")
        return build()

    def patch(self, model, clean, corrupted, *, capture, patch):
        self.calls.append("patch")
        return patch(capture())

    def generate(self, model, prompts, step, *, new_tokens, bounded):
        self.calls.append("generate")
        return torch.cat([step() for _ in range(new_tokens)])

    def generate_patch(self, model, clean, corrupted, *, capture, build_step, new_tokens, bounded):
        self.calls.append("generate_patch")
        value = capture()
        return torch.cat([build_step(value) for _ in range(new_tokens)])

    def train_patch(self, *args, **kwargs):
        self.calls.append("train_patch")
        raise GradientObserved

    def attribute(self, *args, **kwargs):
        self.calls.append("attribute")
        raise GradientObserved

    def vjp_batch(self, *args, **kwargs):
        self.calls.append("vjp_batch")
        raise GradientObserved


def bound_case(methodology, *, backend="hf", family="gpt2", **overrides):
    fn = get_cell(methodology, family, backend)
    # Take every default from the actual cell signature. Branch assertions below execute this cell.
    params = {name: p.default for name, p in inspect.signature(fn).parameters.items()
              if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty}
    params.update(overrides)
    record = describe_case(methodology, params, template=PROTOCOLS[methodology], family=family)
    assert record["protocol_scope"] == "case"
    return fn, params, record["protocol"]


@pytest.mark.parametrize("alpha", [0, 1])
def test_steering_requirements_follow_actual_replacement_write(alpha):
    fn, params, protocol = bound_case("steering", alpha=alpha, layer=0, mode="replace")
    model = model_fixture()
    fn(BackendSpy(), model, ["prompt"], **params)
    wrote = model.transformer.h[0].writes > 0
    assert wrote == ("write" in protocol["operations"]) == (alpha != 0)
    assert protocol["write_components"] == (["block_output"] if wrote else [])


@pytest.mark.parametrize("backend", ["hf", "vllm_async"])
@pytest.mark.parametrize("alpha", [0, 1])
def test_generation_steering_requirements_follow_public_cell_alpha_branch(backend, alpha):
    fn, params, protocol = bound_case("gen_steering", backend=backend, alpha=alpha,
                                      layer=0, new_tokens=2)
    model = model_fixture()
    fn(BackendSpy(), model, ["prompt"], **params)
    writes = model.transformer.h[0].writes
    assert writes == (2 if alpha else 0)
    assert ("write" in protocol["operations"]) == bool(writes)
    assert ("decode_step_write" in protocol["extensions"]) == bool(writes)
    if not writes:
        assert protocol["components"] == ["lm_head"]


@pytest.mark.parametrize("methodology", ["activation_patching", "gen_patching"])
@pytest.mark.parametrize("patch", [False, True])
def test_patch_requirements_follow_single_and_paired_execution(methodology, patch):
    overrides = {"patch": patch, "layer": 0}
    if methodology == "gen_patching":
        overrides["new_tokens"] = 2
    fn, params, protocol = bound_case(methodology, **overrides)
    model, backend = model_fixture(), BackendSpy()
    fn(backend, model, ["clean", "corrupted"], **params)
    wrote = model.transformer.h[0].writes > 0
    assert wrote == patch == ("write" in protocol["operations"])
    assert ("paired_forward" in protocol["capabilities"]) == patch
    assert protocol["data_roles"] == ["base", "counterfactual"]
    assert any("patch" in call for call in backend.calls) == patch
    if not patch and methodology == "gen_patching":
        assert protocol["components"] == ["lm_head"]


@pytest.mark.parametrize("target,component", [
    ("none", None), ("mlp", "mlp_output"), ("attn", "attention_output"),
])
def test_ablation_requirements_follow_actual_target_write(target, component):
    fn, params, protocol = bound_case("ablation", target=target, layer=0)
    model = model_fixture()
    fn(BackendSpy(), model, ["prompt"], **params)
    block = model.transformer.h[0]
    assert block.mlp.writes == int(target == "mlp")
    assert block.attn.writes == int(target == "attn")
    assert protocol["write_components"] == ([component] if component else [])
    assert ("write" in protocol["operations"]) == (component is not None)


def test_hybrid_ablation_keeps_unresolved_component_explicit():
    _, params, protocol = bound_case("ablation", family="nemotron")
    assert params["target"] == "mixer"
    assert protocol["write_components"] is None
    assert protocol["required_capabilities"] is None


@pytest.mark.parametrize("train", [0, 1])
def test_das_requirements_follow_training_backend_call(train):
    fn, params, protocol = bound_case("das", train=train, layer=0, heldout=1, k=1)
    backend = BackendSpy()
    prompts = [("clean", "corrupted", ("a", "b"))] * 2
    if train:
        with pytest.raises(GradientObserved):
            fn(backend, model_fixture(), prompts, **params)
    else:
        fn(backend, model_fixture(), prompts, **params)
    trained = "train_patch" in backend.calls
    assert trained == bool(train) == ("grad" in protocol["operations"])
    assert ("grad" in protocol["capabilities"]) == trained
    assert "write" in protocol["operations"]


@pytest.mark.parametrize("methodology", ["attribution_patching", "jacobian_collect"])
@pytest.mark.parametrize("grad", [False, True])
def test_gradient_requirements_follow_backend_gradient_entrypoints(methodology, grad):
    fn, params, protocol = bound_case(methodology, grad=grad)
    backend = BackendSpy()
    prompts = (("clean", "corrupted", ("a", "b")) if methodology == "attribution_patching"
               else ["prompt"])
    if grad:
        with pytest.raises(GradientObserved):
            fn(backend, model_fixture(), prompts, **params)
    else:
        fn(backend, model_fixture(), prompts, **params)
    observed_grad = any(call in {"attribute", "vjp_batch"} for call in backend.calls)
    assert observed_grad == grad == ("grad" in protocol["operations"])
    assert ("grad" in protocol["capabilities"]) == grad


def test_missing_bound_values_retain_template_scope_without_guessing_defaults():
    template = PROTOCOLS["steering"]
    assert describe_case("steering", {}, template=template, family="gpt2") == {
        "protocol": template.coordinate(), "protocol_scope": "template",
    }


def test_saved_custom_template_is_preserved_for_builtin_method():
    template = replace(PROTOCOLS["steering"], semantic_params=("strength",))
    record = describe_case("steering", {"strength": 0}, template=template, family="gpt2")
    assert record == {"protocol": template.coordinate(), "protocol_scope": "template"}


def test_custom_hook_scope_and_parameter_isolation(monkeypatch):
    monkeypatch.setitem(CASE_DESCRIPTIONS, "custom", lambda *a, **k: None)
    template = InterventionSpec(components=("block_output",), semantic_params=("layers",))

    @case_description("custom", override=True)
    def custom(params, *, template, family):
        assert family == "gpt2"
        params["layers"].append(9)
        return replace(template, operations=("read", "grad"), capabilities=("grad",))

    params = {"layers": [1]}
    record = describe_case("custom", params, template=template, family="gpt2")
    assert params == {"layers": [1]}
    assert record["protocol_scope"] == "case"
    assert "grad" in record["protocol"]["operations"]
    with pytest.raises(ValueError, match="already registered"):
        case_description("custom")(custom)


def test_explicitly_absent_template_stays_absent(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an absent template must not invoke a specialization hook")

    monkeypatch.setitem(CASE_DESCRIPTIONS, "steering", forbidden)
    assert describe_case("steering", {"alpha": 0}, template=None, family="gpt2") == {
        "protocol": None, "protocol_scope": "template",
    }


def test_missing_hook_retains_explicit_template_scope():
    template = InterventionSpec(components=("block_output",))
    assert describe_case("no_hook", {}, template=template, family="gpt2") == {
        "protocol": template.coordinate(), "protocol_scope": "template",
    }


def test_bad_custom_hook_result_is_loud(monkeypatch):
    monkeypatch.setitem(CASE_DESCRIPTIONS, "bad_hook", lambda *a, **k: {})
    with pytest.raises(TypeError, match="InterventionSpec or None"):
        describe_case("bad_hook", {}, template=InterventionSpec(), family="gpt2")


def test_protocol_import_remains_torch_free():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import isb.protocol; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

"""CausaLab-aligned protocol metadata: no model or GPU required."""
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.protocol import CAUSALAB_REFERENCE, COMPONENTS, PROTOCOLS, InterventionSpec  # noqa: E402
from isb.specs import SPECS  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec  # noqa: E402


def test_current_upstream_vocabulary_and_deprecated_alias():
    assert CAUSALAB_REFERENCE["revision"] == "8696e04bfb06a169defe1bf563d8aeef992f85cd"
    assert "attention_value" not in COMPONENTS
    old = InterventionSpec(components=("attention_value",), operations=("read", "write"),
                           write_components=("attention_value",))
    assert old.components == ("attention_premix",)
    assert old.write_components == ("attention_premix",)
    desc = InterventionSpec(components=("attention_value_states", "deltanet_state", "expert_activation"),
                            featurizers=("identity", "pca", "sae", "standardize"),
                            capabilities=("quantized_weights",))
    assert "component:attention_value_states" in desc.required_capabilities()
    assert desc.coordinate()["vocabulary"] == CAUSALAB_REFERENCE


def test_component_requirements_name_write_targets_separately():
    required = PROTOCOLS["ablation"].required_capabilities()
    assert {"component:mlp_output:write", "component:attention_output:write"} <= set(required)
    assert "component:block_output" in required
    assert "component:block_output:write" not in required
    legacy = InterventionSpec(components=("block_output", "lm_head"), operations=("read", "write"))
    assert legacy.required_capabilities() is None
    assert legacy.coordinate()["write_components"] is None


@pytest.mark.parametrize("kwargs", [
    {"components": ("lm_head",), "operations": ("read", "write"), "write_components": ("block_output",)},
    {"components": ("block_output",), "write_components": ("block_output",)},
])
def test_invalid_write_component_descriptions_are_rejected(kwargs):
    with pytest.raises(ValueError, match="write components"):
        InterventionSpec(**kwargs)


def test_every_registered_spec_has_protocol_semantics():
    missing = [name for name, spec in SPECS.items() if spec.protocol is None]
    assert missing == []


def test_protocol_coordinate_is_family_independent_and_method_specific():
    assert (SPECS["logit_lens_gpt2"].protocol.coordinate()
            == SPECS["logit_lens_qwen"].protocol.coordinate())
    assert (SPECS["logit_lens_gpt2"].protocol.coordinate()
            != SPECS["steering_gpt2"].protocol.coordinate())


def test_unknown_protocol_vocabulary_and_unclassified_params_are_loud():
    with pytest.raises(ValueError, match="components"):
        InterventionSpec(components=("made_up_site",))
    with pytest.raises(ValueError, match="featurizers"):
        InterventionSpec(featurizers=("made_up_transform",))
    with pytest.raises(ValueError, match="unclassified"):
        PROTOCOLS["logit_lens"].classify({"mystery": 1})


def test_all_registered_cell_keyword_options_can_be_classified():
    """The built-in templates classify today's registered cell keyword options.

    Custom descriptors and runtime dataset overrides are separately validated at the call boundary.
    """
    import isb.methodologies  # noqa: F401
    from isb.methodologies.registry import CELLS

    for (method, family, backend), fn in CELLS.items():
        if method not in PROTOCOLS:
            continue
        options = {name: param.default for name, param in inspect.signature(fn).parameters.items()
                   if param.kind == inspect.Parameter.KEYWORD_ONLY}
        semantics, realization = PROTOCOLS[method].classify(options)
        assert {**semantics, **realization} == options, (method, family, backend)


def test_spec_params_outside_the_protocol_fail_at_construction():
    spec = SPECS["steering_gpt2"]
    with pytest.raises(ValueError, match="unclassified"):
        replace(spec, tasks=[({"alpah": 0.0}, "typo")])
    with pytest.raises(ValueError, match="unclassified"):
        replace(spec, baseline=BaselineSpec({"alpah": 0.0}))
    with pytest.raises(ValueError, match="unclassified"):
        replace(spec, effect=replace(spec.effect, perturbed_params={"alpah": 6.0}))


@pytest.mark.parametrize("name,params", [
    ("das_gpt2", {"train": 0, "layer": 9, "k": 32, "seed": 7}),
    ("jacobian_lens_gpt2", {"position": "last", "residual": "plain"}),
    ("jacobian_collect_gpt2", {"dim_batch": 32, "skip_first": 4}),
])
def test_supported_custom_cases_keep_exact_cell_parameters(name, params):
    spec = replace(SPECS[name], tasks=[(params, "custom")])
    assert spec.tasks[0].params == params


def test_tuple_tasks_normalize_to_task_specs():
    spec = CellConfig("c", "custom", "gpt2", "repo", [ExecutionRegime("interactive", ["p"])],
                      [({"layer": 1}, "case"), TaskSpec("other", {"layer": 2})], BaselineSpec({}),
                      protocol_absence_reason="fixture")
    assert spec.tasks == [TaskSpec("case", {"layer": 1}), TaskSpec("other", {"layer": 2})]


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_unknown_method_requires_descriptor_or_explicit_reason(reason):
    with pytest.raises(ValueError, match="requires a protocol descriptor"):
        CellConfig("custom", "typo", "gpt2", "repo", [], [], BaselineSpec({}),
                   protocol_absence_reason=reason)


def test_described_method_rejects_absence_reason():
    with pytest.raises(ValueError, match="described methods"):
        replace(SPECS["steering_gpt2"], protocol_absence_reason="skip metadata")

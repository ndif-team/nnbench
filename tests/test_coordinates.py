"""Complete procedure identity and lossless historical coordinate migration."""
import copy
from dataclasses import replace

import pytest

from isb.jobs import contract
from isb.manager.model import comparable
from isb.runs import procedure, run_coordinates, spec_coordinates, upgrade_coordinates
from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime, TaskSpec


def specimen():
    return CellConfig("steering", "steering", "gpt2", "repo",
                      [ExecutionRegime("generation", ["one", "two"], new_tokens=2)],
                      [TaskSpec("steer", {"alpha": 6.0})], BaselineSpec({"alpha": 0}),
                      EffectSpec({"alpha": 0}, {"alpha": 6}), warmup=0, n_trials=1)


def test_spec_coordinates_match_frozen_job_inputs(tmp_path):
    spec = specimen()
    frozen = contract.prepare(spec, tmp_path / "job")
    coords = spec_coordinates(spec, "hf")
    assert coords["schema"] == 3 and coords["identity_complete"]
    assert coords["inputs_sha256"] == frozen["inputs_sha256"]
    assert coords["config"]["baseline"] == {"params": {"alpha": 0}, "label": "baseline"}
    assert procedure(coords) == procedure(spec_coordinates(spec, "vllm_async"))


@pytest.mark.parametrize("change", [
    {"tasks": [TaskSpec("steer", {"alpha": 0})]},
    {"regimes": [ExecutionRegime("generation", ["one", "two"], new_tokens=20)]},
    {"regimes": [ExecutionRegime("generation", ["one", "two"], new_tokens=2, aggregate=False)]},
    {"regimes": [ExecutionRegime("generation", ["one", "two"], new_tokens=2, data_knobs={"alpha": 3})]},
    {"regimes": [ExecutionRegime("generation", ["one", "different"], new_tokens=2)]},
    {"regimes": [ExecutionRegime("generation", ["two", "one"], new_tokens=2)]},
    {"baseline": BaselineSpec({"alpha": 1})},
    {"effect": EffectSpec({"alpha": 0}, {"alpha": 7})},
    {"effect": EffectSpec({"alpha": 0}, {"alpha": 6}, tv_floor=0.8)},
    {"warmup": 2}, {"n_trials": 3}, {"dtype_control": "float16"},
    {"hf_kwargs": {"revision": "other"}}, {"vllm_kwargs": {"trust_remote_code": True}},
])
def test_every_procedure_setting_prevents_accidental_comparison(change):
    spec = specimen()
    base = {"coordinates": spec_coordinates(spec, "hf")}
    variant = {"coordinates": spec_coordinates(replace(spec, **change), "hf")}
    assert not comparable(base, variant)
    assert comparable(base, copy.deepcopy(base))


@pytest.mark.parametrize("old_shape", ["original", "renamed", "schema2"])
def test_historical_details_survive_but_identity_is_unverified(old_shape):
    legacy = {"spec": "steering", "methodology": "steering", "family": "gpt2", "repo": "repo",
              "interface": "hf", "data": ["custom"], "extension": {"items": [1]}}
    regimes = [{"kind": "interactive", "units": 2, "data": "custom",
                "cases": [{"label": "steer", "protocol": {"capabilities": ["write"]}}]}]
    if old_shape == "original":
        legacy.update(workloads=regimes, tasks=["steer"])
    else:
        legacy.update(regimes=regimes, cases=[{"label": "steer", "semantics": {"alpha": 6},
                                               "realization": {"mode": "replace"}}],
                      protocol_template={"mechanisms": ["add_scaled"]})
        if old_shape == "schema2":
            legacy["schema"] = 2
            legacy["cases"][0]["params"] = {"alpha": 7, "mode": "replace"}
    upgraded = upgrade_coordinates(legacy)
    assert upgraded["legacy_coordinates"] == legacy
    assert upgraded["regimes"][0]["cases"] == regimes[0]["cases"]
    assert not upgraded["identity_complete"]
    assert procedure(upgraded) is None
    assert not comparable({"coordinates": upgraded}, {"coordinates": copy.deepcopy(upgraded)})
    if old_shape == "schema2":
        assert upgraded["cases"][0]["params"]["alpha"] == 7
    legacy["extension"]["items"].append(2)
    assert upgraded["legacy_coordinates"]["extension"]["items"] == [1]


def test_coordinate_and_upgrade_snapshots_own_nested_values():
    spec = specimen()
    spec.tasks[0].params["layers"] = [1, 2]
    coords = spec_coordinates(spec, "hf")
    upgraded = upgrade_coordinates(coords)
    spec.tasks[0].params["layers"].append(3)
    coords["cases"][0]["params"]["layers"].append(4)
    assert upgraded["cases"][0]["params"]["layers"] == [1, 2]


@pytest.mark.parametrize("version", [0, 4, 99, "3", 3.0, True])
def test_unknown_coordinate_versions_are_rejected(version):
    coords = spec_coordinates(specimen(), "hf")
    coords["schema"] = version
    with pytest.raises(ValueError, match="schema"):
        upgrade_coordinates(coords)


def test_missing_identity_cannot_be_declared_complete():
    coords = run_coordinates(spec="micro", methodology="constructs", family="gpt2", repo="repo",
                             interface="hf")
    assert not coords["identity_complete"]
    coords["identity_complete"] = True
    with pytest.raises(ValueError, match="exact inputs"):
        upgrade_coordinates(coords)


@pytest.mark.parametrize("change", [
    {"identity_complete": "false"}, {"identity_complete": 1}, {"spec": None},
    {"methodology": None}, {"family": ""}, {"repo": None}, {"cases": []}, {"regimes": []},
    {"cases": [{"label": "task"}]}, {"cases": [{"params": {}}]},
    {"regimes": [{"kind": "interactive"}]}, {"data": None},
])
def test_malformed_current_coordinate_shapes_fail_clearly(change):
    coords = {**spec_coordinates(specimen(), "hf"), **change}
    with pytest.raises(ValueError):
        upgrade_coordinates(coords)
    with pytest.raises(ValueError):
        comparable({"coordinates": coords}, {"coordinates": copy.deepcopy(coords)})


@pytest.mark.parametrize("field,value", [
    ("warmup", None), ("warmup", True), ("warmup", -1), ("n_trials", 0), ("n_trials", "7"),
    ("dtype_control", None), ("hf_kwargs", None), ("vllm_kwargs", []), ("baseline", None),
    ("baseline", {"label": "base", "params": None}), ("effect", {}),
    ("effect", {"baseline_params": {}, "perturbed_params": {}, "tv_floor": float("nan"), "top1_ceiling": 0.5}),
])
def test_malformed_current_controls_cannot_establish_identity(field, value):
    coords = spec_coordinates(specimen(), "hf")
    coords["config"][field] = value
    with pytest.raises(ValueError):
        upgrade_coordinates(coords)


def test_current_completeness_field_is_required():
    coords = spec_coordinates(specimen(), "hf")
    coords.pop("identity_complete")
    with pytest.raises(ValueError, match="boolean"):
        upgrade_coordinates(coords)


def test_coordinates_do_not_bind_or_classify_dataset_overrides():
    from isb.data import DataRef
    from isb.specs import SPECS
    from isb.sweep.spec import spec_with_data

    rebound = spec_with_data(SPECS["logit_lens_gpt2"], DataRef("jlens/poetry", n=2))
    coords = spec_coordinates(rebound, "hf")
    assert "position" in coords["regimes"][0]["data_knobs"]
    assert coords["data"] == ["jlens/poetry"]

"""CausaLab-aligned protocol metadata — no model or GPU required."""
import sys
import inspect
from dataclasses import replace
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.protocol import InterventionSpec, PROTOCOLS, describe_task  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.specs import SPECS  # noqa: E402
from isb.sweep.execute import _run_coordinates  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec  # noqa: E402


def test_task_owns_nested_config_and_returns_independent_snapshots():
    source = {"layers": [1, 2]}
    task = TaskSpec("case", source, {"options": {"order": [3]}})
    source["layers"].append(99)
    params, coordinate = task.params, task.coordinate()
    params["layers"].append(4)
    params["options"]["order"].append(5)
    task.semantics["layers"].append(6)
    assert coordinate["semantics"]["layers"] == [1, 2]
    assert coordinate["realization"]["options"]["order"] == [3]
    assert task.params["layers"] == [1, 2, 6]
    assert task.params["options"]["order"] == [3]


def test_worker_coordinates_are_snapshots_of_frozen_experiment(tmp_path):
    from isb.jobs.contract import prepare, read_experiment, restore_spec

    spec = deepcopy(SPECS["jacobian_lens_qwen35"])
    experiment = prepare(spec, tmp_path / "job")
    worker, _ = restore_spec(tmp_path / "job")
    coords = _run_coordinates(worker, RunConfig(engine=EngineConfig("transformers")))
    before = deepcopy(coords)
    for task in worker.tasks:
        task.semantics["layers"].append(999)
    assert coords == before
    assert read_experiment(tmp_path / "job") == experiment


def test_every_registered_spec_has_protocol_semantics():
    missing = [name for name, spec in SPECS.items() if spec.protocol is None]
    assert missing == []


def test_task_params_are_partitioned_without_changing_cell_call():
    task = SPECS["steering_gpt2"].tasks[0]
    assert task.semantics == {"layer": 8, "target": " Rome", "alpha": 6.0}
    assert task.realization == {"mode": "inplace"}
    assert task.params == {"layer": 8, "target": " Rome", "alpha": 6.0, "mode": "inplace"}

    bounded = SPECS["gen_patching_gpt2"].tasks[0]
    assert bounded.semantics == {"layer": 9}
    assert bounded.realization == {"residual": "plain", "bound": "bounded"}


def test_protocol_marks_nnbench_only_semantics_as_extensions():
    protocol = SPECS["gen_steering_gpt2"].protocol
    assert protocol.position_frames == ("generated", "prompt")
    assert "generate" in protocol.capabilities
    assert "decode_step_write" in protocol.extensions


def test_protocol_coordinate_is_family_independent_and_method_specific():
    assert (SPECS["logit_lens_gpt2"].protocol.coordinate()
            == SPECS["logit_lens_qwen"].protocol.coordinate())
    assert (SPECS["logit_lens_gpt2"].protocol.coordinate()
            != SPECS["steering_gpt2"].protocol.coordinate())


def test_run_coordinates_persist_protocol_and_case_axes():
    spec = SPECS["steering_gpt2"]
    coords = _run_coordinates(spec, RunConfig(engine=EngineConfig("transformers")))
    assert coords["protocol_template"]["mechanisms"] == ["add_scaled"]
    assert coords["cases"][0]["protocol"]["mechanisms"] == ["add_scaled"]
    assert {k: coords["cases"][0][k] for k in ("label", "semantics", "realization")} == spec.tasks[0].coordinate()
    assert coords["regimes"][0]["kind"] == "interactive"


def test_unknown_protocol_vocabulary_and_unclassified_params_are_loud():
    try:
        InterventionSpec(components=("made_up_site",))
        raise AssertionError("unknown component accepted")
    except ValueError as exc:
        assert "components" in str(exc)

    try:
        InterventionSpec(featurizers=("made_up_transform",))
        raise AssertionError("unknown featurizer accepted")
    except ValueError as exc:
        assert "featurizers" in str(exc)

    protocol = SPECS["logit_lens_gpt2"].protocol
    try:
        protocol.classify({"mystery": 1})
        raise AssertionError("unclassified task param accepted")
    except ValueError as exc:
        assert "unclassified" in str(exc)


def test_task_param_namespaces_cannot_overlap():
    try:
        TaskSpec("bad", semantics={"x": 1}, realization={"x": 2})
        raise AssertionError("overlapping namespaces accepted")
    except ValueError as exc:
        assert "both semantic and realization" in str(exc)


def test_explicit_task_must_follow_methodology_parameter_split():
    try:
        CellConfig(
            name="bad", methodology="steering", family="gpt2", repo="repo",
            regimes=[ExecutionRegime("interactive", ["prompt"])],
            tasks=[TaskSpec("bad", semantics={"mode": "inplace"})],
            baseline=BaselineSpec(params={}),
        )
        raise AssertionError("misclassified explicit task accepted")
    except ValueError as exc:
        assert "protocol parameter split" in str(exc)


def test_all_registered_cell_keyword_options_can_be_classified():
    import isb.methodologies  # noqa: F401
    from isb.methodologies.registry import CELLS

    for (method, family, backend), fn in CELLS.items():
        if method not in PROTOCOLS:
            continue
        options = {name: param.default for name, param in inspect.signature(fn).parameters.items()
                   if param.kind == inspect.Parameter.KEYWORD_ONLY}
        semantics, realization = PROTOCOLS[method].classify(options)
        assert {**semantics, **realization} == options, (method, family, backend)


@pytest.mark.parametrize("name,params", [
    ("das_gpt2", {"train": 0, "layer": 9, "k": 32, "seed": 7}),
    ("jacobian_lens_gpt2", {"position": "last", "residual": "plain"}),
    ("jacobian_collect_gpt2", {"dim_batch": 32, "skip_first": 4}),
])
def test_supported_custom_cases_keep_exact_cell_parameters(name, params):
    spec = replace(SPECS[name], tasks=[(params, "custom")])
    assert spec.tasks[0].params == params


@pytest.mark.parametrize("method,params,grad,write,paired", [
    ("das", {}, False, True, True),
    ("das", {"train": 0}, False, True, True),
    ("das", {"train": 24}, True, True, True),
    ("attribution_patching", {}, True, False, False),
    ("attribution_patching", {"grad": False}, False, False, False),
    ("jacobian_collect", {}, True, False, False),
    ("jacobian_collect", {"grad": False}, False, False, False),
    ("activation_patching", {"patch": False}, False, False, False),
    ("activation_patching", {}, False, True, True),
    ("gen_patching", {"patch": False}, False, False, False),
    ("steering", {"alpha": 0}, False, False, False),
    # This cell still assigns the output even when the numerical delta is zero.
    ("gen_steering", {"alpha": 0}, False, True, False),
    ("ablation", {"target": "none"}, False, False, False),
])
def test_task_requirements_follow_cell_branches(method, params, grad, write, paired):
    template = PROTOCOLS[method]
    before = template.coordinate()
    desc = describe_task(method, params, template=template, family="gpt2")
    assert ("grad" in desc.operations) == grad
    assert ("grad" in desc.capabilities) == grad
    assert ("write" in desc.operations) == write
    assert bool(desc.mechanisms) == write
    assert ("paired_forward" in desc.capabilities) == paired
    assert template.coordinate() == before
    if method.startswith("gen_"):
        assert "generate" in desc.capabilities


def test_regime_coordinates_capture_execution_and_effective_requirements():
    run = RunConfig(engine=EngineConfig("transformers"))
    spec = SPECS["gen_patching_gpt2"]
    short = replace(spec, regimes=[replace(spec.regimes[0], new_tokens=2)])
    long = replace(spec, regimes=[replace(spec.regimes[0], new_tokens=20)])
    a, b = _run_coordinates(short, run), _run_coordinates(long, run)
    assert a != b
    assert a["regimes"][0]["new_tokens"] == 2
    assert b["regimes"][0]["new_tokens"] == 20
    assert a["regimes"][0]["cases"][0]["semantics"]["new_tokens"] == 2

    das = SPECS["das_gpt2"]
    bound = replace(das, tasks=[({}, "dataset default"), ({"train": 0}, "explicit apply")],
                    regimes=[replace(das.regimes[0], data_knobs={"train": 3})])
    coords = _run_coordinates(bound, run)
    regime = coords["regimes"][0]
    assert regime["aggregate"] is False
    assert regime["data_knobs"] == {"train": 3}
    assert "grad" in regime["cases"][0]["protocol"]["capabilities"]
    assert "grad" not in regime["cases"][1]["protocol"]["capabilities"]
    assert regime["cases"][1]["semantics"]["train"] == 0


def test_custom_method_keeps_explicit_realization_without_protocol():
    spec = CellConfig(
        "custom", "custom", "gpt2", "repo",
        [ExecutionRegime("interactive", ["p"], data_knobs={"scale": 2})],
        [TaskSpec("case", {"layer": 1}, {"mode": "replace"})], BaselineSpec({}),
        protocol_absence_reason="Experimental method with author-supplied parameter split")
    coords = _run_coordinates(spec, RunConfig(engine=EngineConfig("transformers")))
    assert coords["cases"][0] == spec.tasks[0].coordinate()
    bound = coords["regimes"][0]["cases"][0]
    assert bound["semantics"] == {"layer": 1, "scale": 2}
    assert bound["realization"] == {"mode": "replace"}
    assert coords["protocol_coverage"] == {
        "status": "undescribed", "reason": spec.protocol_absence_reason}


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_unknown_method_requires_descriptor_or_explicit_reason(reason):
    with pytest.raises(ValueError, match="requires a protocol descriptor"):
        CellConfig("custom", "typo", "gpt2", "repo", [], [], BaselineSpec({}),
                   protocol_absence_reason=reason)


def test_described_method_rejects_absence_reason():
    with pytest.raises(ValueError, match="described methods"):
        replace(SPECS["steering_gpt2"], protocol_absence_reason="skip metadata")

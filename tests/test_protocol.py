"""CausaLab-aligned protocol metadata — no model or GPU required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.protocol import InterventionSpec  # noqa: E402
from isb.runs import EngineConfig, RunConfig  # noqa: E402
from isb.specs import SPECS  # noqa: E402
from isb.sweep.execute import _run_coordinates  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec  # noqa: E402


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
    assert coords["protocol"]["mechanisms"] == ["add_scaled"]
    assert coords["cases"][0] == spec.tasks[0].coordinate()
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

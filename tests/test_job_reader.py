"""Current job boundaries, frozen descriptors, and structured restoration failures."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from isb.jobs import contract, worker
from isb.protocol import InterventionSpec
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec


def fixture_job(tmp_path):
    spec = CellConfig(
        "saved", "steering", "gpt2", "repo",
        [ExecutionRegime("interactive", [("clean", "corrupt")], aggregate=False)],
        [TaskSpec("case", {"saved_option": [1, 2], "mode": "saved"})],
        BaselineSpec({}), warmup=0, n_trials=1,
        protocol=InterventionSpec(semantic_params=("saved_option",), realization_params=("mode",)))
    job = tmp_path / "job"
    return job, contract.prepare(spec, job)


def resign(job, experiment):
    experiment.pop("id", None)
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(job / "experiment.json", experiment)


def test_frozen_builtin_override_and_identity_survive(tmp_path):
    job, original = fixture_job(tmp_path)
    spec, experiment = contract.restore_spec(job)
    assert experiment == original
    assert spec.protocol_source == "restored"
    assert spec.protocol.semantic_params == ("saved_option",)
    assert spec.protocol.realization_params == ("mode",)
    assert spec.tasks[0].params == {"saved_option": [1, 2], "mode": "saved"}
    assert spec.regimes[0].prompts == [("clean", "corrupt")]
    assert spec.protocol_coverage() == {"status": "described"}
    assert contract.expected_cells(experiment) == {("interactive", "case")}
    assert "protocol_source" not in contract.prepare(spec, tmp_path / "rewritten")["spec"]
    assert contract.read_experiment(job) == original


def test_restoration_defers_parameter_classification_for_saved_calls(tmp_path):
    job, experiment = fixture_job(tmp_path)
    experiment["spec"]["protocol"]["semantic_params"] = contract.pack(("different",))
    resign(job, experiment)
    restored, _ = contract.restore_spec(job)
    assert restored.tasks[0].params["saved_option"] == [1, 2]
    with pytest.raises(ValueError, match="unclassified"):
        restored.protocol.classify(restored.tasks[0].params)
    with pytest.raises(ValueError, match="unclassified"):
        replace(restored, protocol_source="authored")


@pytest.mark.parametrize("version", [0, 1, 3, True, 2.0, "2"])
def test_noncurrent_versions_are_rejected_at_every_boundary(tmp_path, version):
    job, experiment = fixture_job(tmp_path)
    experiment["version"] = version
    resign(job, experiment)
    with pytest.raises(ValueError, match="version"):
        contract.restore_spec(job)
    with pytest.raises(ValueError, match="version"):
        contract.wire_spec(experiment)


@pytest.mark.parametrize("damage,known_identity", [
    ("bad_json", False), ("bad_checksum", False), ("non_object", False),
    ("input_checksum", True), ("bad_spec", True), ("duplicate_task", True),
])
def test_worker_reports_restoration_errors_atomically(tmp_path, monkeypatch, damage, known_identity):
    job, experiment = fixture_job(tmp_path)
    monkeypatch.setenv("ISB_BACKEND", "independent-package")
    if damage == "bad_json":
        (job / "experiment.json").write_text("{")
    elif damage == "bad_checksum":
        experiment["id"] = "0" * 64
        contract.write_json(job / "experiment.json", experiment)
    elif damage == "non_object":
        contract.write_json(job / "experiment.json", [])
    elif damage == "input_checksum":
        (job / "inputs.jsonl").write_text("tampered\n")
    elif damage == "bad_spec":
        experiment["spec"]["surprise_field"] = True
        resign(job, experiment)
    else:
        experiment["spec"]["tasks"].append(experiment["spec"]["tasks"][0])
        resign(job, experiment)

    def forbidden(spec):
        raise AssertionError("backend creation should never be reached")

    output = tmp_path / "output"
    with pytest.raises(Exception) as caught:
        worker.main(forbidden, job, output)
    assert not isinstance(caught.value, AssertionError)
    assert sorted(p.name for p in output.iterdir()) == ["result.json"]
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["backend"] == "independent-package"
    assert result["error"]
    assert ("experiment_id" in result) == known_identity
    if known_identity:
        assert result["experiment_id"] == experiment["id"]
        assert result["inputs_sha256"] == experiment["inputs_sha256"]


def test_v2_rejects_missing_saved_protocol_instead_of_guessing(tmp_path):
    job, _ = fixture_job(tmp_path)
    spec, _ = contract.restore_spec(job)
    spec.protocol = InterventionSpec(semantic_params=("saved_option",), realization_params=("mode",))
    new_job = tmp_path / "v2"
    experiment = contract.prepare(spec, new_job)
    experiment["spec"]["protocol"] = None
    resign(new_job, experiment)
    with pytest.raises(ValueError, match="requires a protocol"):
        contract.restore_spec(new_job)


@pytest.mark.parametrize("stage", ["execute", "completion"])
def test_worker_failure_cleans_its_staging_and_published_artifact(tmp_path, monkeypatch, stage):
    from types import SimpleNamespace

    import torch
    import isb.runs
    import isb.sweep.execute as execute

    job, experiment = fixture_job(tmp_path)
    monkeypatch.setenv("ISB_BACKEND", "independent-package")
    monkeypatch.setattr(isb.runs, "resolve_provenance", lambda run: {})

    def fake_execute(spec, run, output, tag, **kwargs):
        torch.save({"outputs": {("interactive", "case"): torch.ones(1)},
                    "provenance": kwargs["provenance"]}, Path(output) / "result.pt")
        if stage == "execute":
            raise ValueError("fixture execution failed")

    def fail_completion(path, result):
        if result["status"] == "completed":
            raise ValueError("fixture completion failed")
        return contract.write_json(path, result)

    monkeypatch.setattr(execute, "execute_run", fake_execute)
    monkeypatch.setattr(worker, "write_json", fail_completion)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match=f"fixture {stage if stage == 'completion' else 'execution'} failed"):
        worker.main(lambda spec: (SimpleNamespace(name="test-interface"), None), job, output)
    assert sorted(p.name for p in output.iterdir()) == ["result.json"]
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "failed"
    assert result["experiment_id"] == experiment["id"]

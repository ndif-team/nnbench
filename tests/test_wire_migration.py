"""Historical job compatibility and structured restoration failures.

The checked fixtures were emitted by main's actual v1 writer at 119a1ba using historical
task dataclasses. Their fixed checksums protect the test from accidentally following the new
writer's shape. Tests modify/re-sign a copy only when exercising invalid or earlier records.
"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from isb.jobs import contract, worker
from isb.protocol import InterventionSpec


def fixture_job(tmp_path, name="described_builtin"):
    fixture = json.loads((Path(__file__).parent / "fixtures/v1-experiments.json").read_text())["fixtures"][name]
    job = tmp_path / "job"
    job.mkdir()
    (job / "inputs.jsonl").write_text(fixture["inputs"])
    contract.write_json(job / "experiment.json", fixture["experiment"])
    return job, fixture["experiment"]


def resign(job, experiment):
    experiment.pop("id", None)
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(job / "experiment.json", experiment)


def test_main_v1_frozen_builtin_override_and_identity_survive(tmp_path):
    job, original = fixture_job(tmp_path)
    spec, experiment = contract.restore_spec(job)
    assert experiment == original
    assert spec.protocol_source == "restored"
    assert spec.protocol.semantic_params == ("old_option",)
    assert spec.protocol.realization_params == ("mode",)
    assert spec.tasks[0].params == {"old_option": [1, 2], "mode": "saved"}
    assert spec.regimes[0].prompts == [("clean", "corrupt")]
    assert spec.protocol_coverage() == {"status": "described"}
    assert contract.expected_cells(experiment) == {("interactive", "case")}
    assert "protocol_source" not in contract.prepare(spec, tmp_path / "rewritten")["spec"]
    assert contract.read_experiment(job) == original


def test_main_v1_explicit_coverage_reason_survives(tmp_path):
    job, original = fixture_job(tmp_path, "explicit_opt_out")
    spec, experiment = contract.restore_spec(job)
    assert experiment == original
    assert spec.protocol is None
    assert spec.protocol_source == "restored"
    assert spec.protocol_coverage() == original["description"]["protocol_coverage"]


@pytest.mark.parametrize("name,status", [("described_builtin", "described"),
                                         ("explicit_opt_out", "legacy")])
def test_early_v1_description_without_coverage_keeps_only_known_metadata(tmp_path, name, status):
    job, experiment = fixture_job(tmp_path, name)
    experiment["description"].pop("protocol_coverage")
    resign(job, experiment)
    spec, original = contract.restore_spec(job)
    assert original == experiment
    assert spec.protocol_coverage()["status"] == status
    assert (spec.protocol is not None) == (status == "described")


@pytest.mark.parametrize("name", ["described_builtin", "explicit_opt_out"])
def test_original_v1_keeps_unknown_metadata_for_every_method(tmp_path, name):
    job, experiment = fixture_job(tmp_path, name)
    experiment.pop("description")
    resign(job, experiment)
    spec, original = contract.restore_spec(job)
    assert original == experiment
    assert spec.protocol is None
    assert spec.protocol_source == "legacy"
    assert spec.protocol_coverage() == {"status": "legacy", "reason": contract.LEGACY_PROTOCOL_REASON}
    assert spec.tasks[0].params["old_option"] == [1, 2]
    rewritten = tmp_path / "v2"
    contract.prepare(spec, rewritten)
    assert contract.restore_spec(rewritten)[0].protocol_coverage() == spec.protocol_coverage()


@pytest.mark.parametrize("change,match", [
    ("value", "does not match"), ("label", "does not match"),
    ("overlap", "overlap"), ("coverage", "coverage"), ("null_coverage", "coverage"),
])
def test_inconsistent_v1_descriptions_fail_clearly(tmp_path, change, match):
    job, experiment = fixture_job(tmp_path)
    description = experiment["description"]
    if change == "value":
        description["tasks"][0]["semantics"]["old_option"] = [3]
    elif change == "label":
        description["tasks"][0]["label"] = "other"
    elif change == "overlap":
        description["tasks"][0]["realization"]["old_option"] = [1, 2]
    elif change == "coverage":
        description["protocol_coverage"] = {"status": "undescribed", "reason": "contradiction"}
    else:
        description["protocol_coverage"] = None
    resign(job, experiment)
    with pytest.raises(ValueError, match=match):
        contract.restore_spec(job)


def test_restoration_defers_parameter_classification_for_saved_calls(tmp_path):
    job, experiment = fixture_job(tmp_path)
    experiment["description"]["protocol_template"]["semantic_params"] = contract.pack(("different",))
    resign(job, experiment)
    restored, _ = contract.restore_spec(job)
    assert restored.tasks[0].params["old_option"] == [1, 2]
    with pytest.raises(ValueError, match="unclassified"):
        restored.protocol.classify(restored.tasks[0].params)
    with pytest.raises(ValueError, match="unclassified"):
        replace(restored, protocol_source="authored")


@pytest.mark.parametrize("version", [0, 3, True])
def test_unknown_versions_are_rejected_before_migration(tmp_path, version):
    job, experiment = fixture_job(tmp_path)
    experiment["version"] = version
    resign(job, experiment)
    with pytest.raises(ValueError, match="version"):
        contract.restore_spec(job)
    with pytest.raises(ValueError, match="version"):
        contract.wire_spec(experiment)


@pytest.mark.parametrize("damage,known_identity", [
    ("bad_json", False), ("bad_checksum", False), ("non_object", False),
    ("input_checksum", True), ("bad_spec", True), ("duplicate_description", True),
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
        experiment["description"]["tasks"][0]["label"] = "mismatch"
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
    spec.protocol = InterventionSpec(semantic_params=("old_option",), realization_params=("mode",))
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

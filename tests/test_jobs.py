"""Runner contracts and failure modes; no Docker daemon or inference packages required."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from isb.jobs import contract, local
from isb.jobs.cli import _specs, parser
from isb.jobs.worker import merge_requirements
from isb.sweep.spec import BaselineSpec, CellConfig, Workload


def specimen():
    return CellConfig("tiny", "logit_lens", "gpt2", "test/model",
                      [Workload("interactive", ["one", "two"])],
                      [({"layers": (1, 2)}, "layers")], BaselineSpec({}), warmup=0, n_trials=1)


def completed(output, experiment, backend):
    (output / "result.pt").write_bytes(b"a tensor artifact")
    contract.write_json(output / "result.json", {
        "version": 1, "status": "completed", "backend": backend,
        "experiment_id": experiment["id"], "inputs_sha256": experiment["inputs_sha256"],
        "outputs_sha256": contract.file_digest(output / "result.pt"),
        "cells": [{"workload": w, "label": label, "state": "RAN"}
                  for w, label in sorted(contract.expected_cells(experiment))],
    })


def test_inputs_roundtrip_preserves_pairs_and_dataset_level_work(tmp_path):
    units = [("clean", "corrupt", ("right", "wrong")), ("c2", "b2", ("r2", "w2"))]
    spec = replace(specimen(), workloads=[Workload("interactive", units, aggregate=False)])
    experiment = contract.prepare(spec, tmp_path / "job", seed=8)
    restored, reread = contract.restore_spec(tmp_path / "job")
    assert restored == spec
    assert reread == experiment
    assert not restored.workloads[0].aggregate


def test_identity_changes_with_count_params_and_seed(tmp_path):
    spec = specimen()
    variants = [spec, replace(spec, workloads=[Workload("interactive", ["one"])]),
                replace(spec, tasks=[({"layers": [3]}, "layers")])]
    identities = {contract.prepare(s, tmp_path / str(i))["id"] for i, s in enumerate(variants)}
    identities.add(contract.prepare(spec, tmp_path / "seed", seed=1)["id"])
    assert len(identities) == 4


def test_experiment_integrity_and_no_overwrite(tmp_path):
    directory = tmp_path / "job"
    contract.prepare(specimen(), directory)
    with pytest.raises(FileExistsError):
        contract.prepare(specimen(), directory)
    (directory / "inputs.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="input checksum"):
        contract.read_experiment(directory)


@pytest.mark.parametrize("change", [
    {"workloads": [Workload("interactive", [])]},
    {"workloads": [Workload("interactive", ["a"]), Workload("interactive", ["b"])]},
    {"tasks": [({}, "same"), ({}, "same")]}, {"n_trials": 0},
])
def test_reject_ambiguous_or_empty_experiment(tmp_path, change):
    with pytest.raises(ValueError):
        contract.prepare(replace(specimen(), **change), tmp_path / "job")


def test_result_must_account_for_all_cells_and_match_inputs(tmp_path):
    experiment = contract.prepare(specimen(), tmp_path / "job")
    output = tmp_path / "output"
    output.mkdir()
    completed(output, experiment, "example")
    assert contract.validate_result(output, experiment, "example")["status"] == "completed"
    with pytest.raises(ValueError):
        contract.validate_result(output, experiment, "different")
    result = json.loads((output / "result.json").read_text())
    result["cells"] = []
    contract.write_json(output / "result.json", result)
    with pytest.raises(ValueError, match="every requested cell"):
        contract.validate_result(output, experiment, "example")
    completed(output, experiment, "example")
    (output / "result.pt").write_bytes(b"partial")
    with pytest.raises(ValueError, match="corrupted"):
        contract.validate_result(output, experiment, "example")


def test_discovery_is_directory_based_and_rejects_escape(tmp_path):
    backend = tmp_path / "backends" / "third-party"
    backend.mkdir(parents=True)
    (backend / "compose.yml").write_text("services: {runner: {image: example}}")
    assert local.discover(tmp_path) == ["third-party"]
    assert local.backend_file("third-party", tmp_path) == backend / "compose.yml"
    for name in ("../third-party", "/tmp", "missing", "third-party/child"):
        with pytest.raises(ValueError):
            local.backend_file(name, tmp_path)


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    backend = tmp_path / "backends" / "third-party"
    backend.mkdir(parents=True)
    (backend / "compose.yml").write_text("services: {runner: {image: example}}")
    experiment = contract.prepare(specimen(), tmp_path / "job")
    calls = []
    behavior = {"kind": "success"}

    def run(argv, **kwargs):
        calls.append(argv)
        if "config" in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"services": {"runner": {}}}))
        if "run" in argv:
            if behavior["kind"] == "timeout":
                raise subprocess.TimeoutExpired(argv, 1)
            if behavior["kind"] == "cancel":
                raise KeyboardInterrupt
            if behavior["kind"] == "crash":
                return subprocess.CompletedProcess(argv, 17)
            if behavior["kind"] != "missing":
                completed(Path(kwargs["env"]["ISB_OUTPUT_DIR"]), experiment, "third-party")
        if "inspect" in argv:
            return subprocess.CompletedProcess(argv, 0, "sha256:actual-image\n")
        return subprocess.CompletedProcess(argv, 0, "")

    monkeypatch.setattr(local.subprocess, "run", run)
    return experiment, calls, behavior


@pytest.mark.parametrize("kind,status", [("success", "completed"), ("missing", "failed"),
                                        ("timeout", "failed"), ("crash", "failed")])
def test_lifecycle_always_cleans_and_validates(tmp_path, fake_docker, kind, status):
    experiment, calls, behavior = fake_docker
    behavior["kind"] = kind
    record = local.run_job("third-party", tmp_path / "job", experiment, tmp_path / "output", root=tmp_path)
    assert record["status"] == status
    assert record["image_id"] == "sha256:actual-image"
    assert any("down" in cmd for cmd in calls)
    assert not any("--volumes" in cmd for cmd in calls)
    assert "third-party" not in next(cmd for cmd in calls if "run" in cmd)[-1:]
    assert json.loads((tmp_path / "output" / "execution.json").read_text())["status"] == status


def test_cancel_cleans_before_propagating(tmp_path, fake_docker):
    experiment, calls, behavior = fake_docker
    behavior["kind"] = "cancel"
    with pytest.raises(KeyboardInterrupt):
        local.run_job("third-party", tmp_path / "job", experiment, tmp_path / "output", root=tmp_path)
    assert any("down" in cmd for cmd in calls)
    assert json.loads((tmp_path / "output" / "execution.json").read_text())["status"] == "cancelled"


def test_source_changes_invalidate_a_completed_container(tmp_path, fake_docker, monkeypatch):
    experiment, calls, _ = fake_docker
    identities = iter([{"source_sha256": "before"}, {"source_sha256": "after"}])
    monkeypatch.setattr(local, "source_identity", lambda root: next(identities))
    record = local.run_job("third-party", tmp_path / "job", experiment, tmp_path / "output", root=tmp_path)
    assert record["status"] == "failed"
    assert "source changed" in record["error"]
    assert any("down" in cmd for cmd in calls)


@pytest.mark.parametrize("service", [{"ports": ["6677:6677"]}, {"container_name": "shared"},
                                     {"network_mode": "host"}])
def test_backend_services_must_be_job_isolated(tmp_path, fake_docker, monkeypatch, service):
    monkeypatch.setattr(local.subprocess, "run", lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 0, json.dumps({"services": {"runner": service}})))
    with pytest.raises(ValueError):
        local.configuration("third-party", tmp_path)


def test_launcher_and_specs_do_not_import_torch():
    code = "import isb.jobs.cli, isb.specs, sys; assert 'torch' not in sys.modules; assert 'nnsight' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=local.ROOT)


def test_requirement_conflicts_are_not_silent():
    assert merge_requirements({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}
    with pytest.raises(ValueError, match="conflicts"):
        merge_requirements({"x": 1}, {"x": 2})


def test_cli_uses_names_and_rejects_nonpositive_counts():
    args = parser().parse_args(["run", "--spec", "logit_lens_gpt2", "--backends", "third-party"])
    assert args.backends == ["third-party"]
    with pytest.raises(ValueError, match="positive"):
        _specs(["logit_lens_gpt2"], "factual:0")

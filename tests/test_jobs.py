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
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec


def specimen():
    return CellConfig("tiny", "logit_lens", "gpt2", "test/model",
                      [ExecutionRegime("interactive", ["one", "two"])],
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
    spec = replace(specimen(), regimes=[ExecutionRegime("interactive", units, aggregate=False)])
    experiment = contract.prepare(spec, tmp_path / "job", seed=8)
    restored, reread = contract.restore_spec(tmp_path / "job")
    assert restored == spec
    assert reread == experiment
    assert not restored.regimes[0].aggregate


def test_protocol_metadata_roundtrips_without_changing_v1_execution_fields(tmp_path):
    from isb.protocol import InterventionSpec

    template = InterventionSpec(semantic_params=("layer",), realization_params=("mode",))
    spec = CellConfig("custom", "custom", "gpt2", "repo",
                      [ExecutionRegime("generation", [("clean", "corrupt")], new_tokens=5,
                                       aggregate=False, data_knobs={"layer": 3})],
                      [TaskSpec("case", {"layer": 4}, {"mode": "replace"})], BaselineSpec({}),
                      protocol=template)
    experiment = contract.prepare(spec, tmp_path / "job")
    wire = contract.unpack(experiment["spec"])
    assert "regimes" not in wire and "protocol" not in wire
    assert wire["tasks"] == [({"layer": 4, "mode": "replace"}, "case")]
    assert contract.expected_cells(experiment) == {("generation", "case")}
    restored, _ = contract.restore_spec(tmp_path / "job")
    assert restored == spec
    changed = replace(spec, protocol=replace(template, components=("block_output",)))
    assert contract.prepare(changed, tmp_path / "changed")["id"] != experiment["id"]


def test_legacy_descriptor_without_write_targets_restores_without_guessing(tmp_path):
    from isb.protocol import InterventionSpec

    template = InterventionSpec(components=("attention_premix",), operations=("read", "write"),
                                write_components=("attention_premix",), semantic_params=("layers",))
    spec = replace(specimen(), protocol=template)
    experiment = contract.prepare(spec, tmp_path / "job")
    old_template = experiment["description"]["protocol_template"]
    old_template.pop("write_components")
    old_template["components"] = contract.pack(("attention_value",))
    experiment.pop("id")
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(tmp_path / "job" / "experiment.json", experiment)
    restored, original = contract.restore_spec(tmp_path / "job")
    assert restored.protocol.components == ("attention_premix",)
    assert restored.protocol.required_capabilities() is None
    assert original == experiment


def test_legacy_v1_experiment_without_description_still_restores(tmp_path):
    experiment = contract.prepare(specimen(), tmp_path / "job")
    experiment.pop("description")
    experiment.pop("id")
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(tmp_path / "job" / "experiment.json", experiment)
    restored, _ = contract.restore_spec(tmp_path / "job")
    assert restored == specimen()


def test_explicit_protocol_opt_out_roundtrips_and_is_identity_covered(tmp_path):
    spec = replace(specimen(), methodology="custom", protocol=None,
                   protocol_absence_reason="Experimental custom method")
    experiment = contract.prepare(spec, tmp_path / "job")
    assert "protocol_absence_reason" not in contract.unpack(experiment["spec"])
    assert experiment["description"]["protocol_coverage"] == spec.protocol_coverage()
    assert contract.restore_spec(tmp_path / "job")[0] == spec
    changed = replace(spec, protocol_absence_reason="Revised explanation")
    assert contract.prepare(changed, tmp_path / "changed")["id"] != experiment["id"]


@pytest.mark.parametrize("has_description", [False, True])
def test_legacy_unknown_method_records_missing_metadata_explicitly(tmp_path, has_description):
    spec = replace(specimen(), methodology="custom", protocol=None,
                   protocol_absence_reason="Experimental fixture")
    experiment = contract.prepare(spec, tmp_path / "job")
    if has_description:
        experiment["description"].pop("protocol_coverage")
    else:
        experiment.pop("description")
    experiment.pop("id")
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(tmp_path / "job" / "experiment.json", experiment)
    restored, _ = contract.restore_spec(tmp_path / "job")
    assert restored.protocol is None
    assert restored.protocol_coverage() == {
        "status": "undescribed", "reason": "Legacy experiment predates explicit protocol coverage"}


@pytest.mark.parametrize("coverage", [None, {}, {"status": "invalid"},
                                     {"status": "undescribed", "reason": "skip"}])
def test_inconsistent_protocol_coverage_is_rejected(tmp_path, coverage):
    experiment = contract.prepare(specimen(), tmp_path / "job")
    experiment["description"]["protocol_coverage"] = coverage
    experiment.pop("id")
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(tmp_path / "job" / "experiment.json", experiment)
    with pytest.raises(ValueError):
        contract.restore_spec(tmp_path / "job")


def test_prepare_rechecks_protocol_coverage_after_spec_edit(tmp_path):
    spec = specimen()
    spec.protocol = None
    with pytest.raises(ValueError, match="requires a protocol"):
        contract.prepare(spec, tmp_path / "job")
    assert not (tmp_path / "job").exists()


def test_description_cannot_disagree_with_executable_tasks(tmp_path):
    experiment = contract.prepare(specimen(), tmp_path / "job")
    experiment["description"]["tasks"][0]["label"] = "different"
    experiment.pop("id")
    experiment["id"] = contract.digest(contract.canonical(experiment))
    contract.write_json(tmp_path / "job" / "experiment.json", experiment)
    with pytest.raises(ValueError, match="description does not match"):
        contract.restore_spec(tmp_path / "job")


def test_container_worker_preserves_case_requirements_and_backend_identity(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    import isb.runs
    import isb.sweep.execute as execute
    from isb.jobs import worker
    from isb.runs import EngineConfig, RunConfig

    spec = CellConfig("das", "das", "gpt2", "repo",
                      [ExecutionRegime("interactive", [("clean", "corrupt")], aggregate=False)],
                      [({"train": 0}, "apply"), ({"train": 24}, "train")], BaselineSpec({"train": 0}),
                      warmup=0, n_trials=1)
    experiment = contract.prepare(spec, tmp_path / "job")
    seen = []

    class Backend:
        name = "custom-interface"

        def load(self, repo):
            return SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"),
                                   tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}))

        def teardown(self, model):
            seen.append("teardown")

    def get_cell(method, family, interface):
        assert interface == "custom-interface"
        def cell(be, model, prompts, **params):
            seen.append(params["train"])
            return torch.ones(1, 2)
        return cell

    def forbidden(*args, **kwargs):
        raise AssertionError("worker must use the container-supplied backend")

    monkeypatch.setenv("ISB_BACKEND", "independent-package")
    monkeypatch.setattr(execute, "get_cell", get_cell)
    monkeypatch.setattr(execute, "make_backend", forbidden)
    monkeypatch.setattr(isb.runs, "resolve_provenance", lambda run: {})
    worker.main(lambda restored: (Backend(), RunConfig(engine=EngineConfig("transformers"))),
                job=tmp_path / "job", output=tmp_path / "output")
    result = contract.validate_result(tmp_path / "output", experiment, "independent-package")
    assert 0 in seen and 24 in seen and seen[-1] == "teardown"
    assert all(cell["state"] == "RAN" for cell in result["cells"])
    prov = result["provenance"]
    assert prov["model_identity"]["revision"] == "revision"
    coords = prov["coordinates"]
    assert coords["interface"] == "custom-interface"
    cases = coords["regimes"][0]["cases"]
    assert "grad" not in cases[0]["protocol"]["capabilities"]
    assert "grad" in cases[1]["protocol"]["capabilities"]


def test_identity_changes_with_count_params_and_seed(tmp_path):
    spec = specimen()
    variants = [spec, replace(spec, regimes=[ExecutionRegime("interactive", ["one"])]),
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
    {"regimes": [ExecutionRegime("interactive", [])]},
    {"regimes": [ExecutionRegime("interactive", ["a"]), ExecutionRegime("interactive", ["b"])]},
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

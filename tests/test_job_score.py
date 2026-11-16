"""Artifact comparison is explicit and rejects mismatched or incomplete experiments."""
import json

import pytest
import torch

from isb.jobs import contract, schema
from isb.jobs.score import score_experiment
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime


def artifacts(tmp_path, *, batched=False):
    spec = CellConfig("tiny", "logit_lens", "gpt2", "test/model",
                      [ExecutionRegime("batched" if batched else "interactive", ["a"])],
                      [({}, "task")], BaselineSpec({}), warmup=0, n_trials=1)
    directory = tmp_path / "experiment"
    experiment = contract.prepare(spec, directory)
    key = ("batched" if batched else "interactive", "task")
    for backend in ("reference-name", "candidate-name"):
        output = directory / backend
        output.mkdir()
        identity = {"version": schema.WIRE_VERSION, "backend": backend, "experiment_id": experiment["id"],
                    "inputs_sha256": experiment["inputs_sha256"]}
        provenance = {"job": identity, "model_identity": {
            "repo": "test/model", "revision": "abc", "vocab_sha256": "def", "vocab_size": 8}}
        value = torch.tensor([[10., 0., 0., 0., 0., 0., 0., 0.]])
        torch.save({"outputs": {key: value, ("batched_perprompt", "task"): value},
                    "provenance": provenance}, output / "result.pt")
        contract.write_json(output / "result.json", {**identity, "status": "completed",
            "cells": [{"regime": key[0], "label": "task", "state": "RAN"}],
            "outputs_sha256": contract.file_digest(output / "result.pt")})
        contract.write_json(output / "execution.json", {"status": "completed", "source": {"commit": "abc"}})
    return directory, key


def change(directory, backend, fn):
    output = directory / backend
    artifact = torch.load(output / "result.pt", weights_only=False)
    fn(artifact)
    torch.save(artifact, output / "result.pt")
    result = json.loads((output / "result.json").read_text())
    result["outputs_sha256"] = contract.file_digest(output / "result.pt")
    contract.write_json(output / "result.json", result)


def candidate(directory, **kwargs):
    return score_experiment(directory, ["reference-name", "candidate-name"],
                            reference="reference-name", **kwargs)["cells"][-1]


def test_policy_does_not_depend_on_backend_names(tmp_path):
    directory, _ = artifacts(tmp_path)
    assert candidate(directory)["state"] == "SUPPORTED"
    assert candidate(directory, comparison="equivalence")["state"] == "EQUIVALENT"


def test_a_methodology_check_decides_the_verdict(tmp_path, monkeypatch):
    from isb.oracle import checks

    directory, key = artifacts(tmp_path)
    seen = []
    monkeypatch.setattr(checks, "CHECKS", {})

    @checks.check_of("logit_lens")
    def never(expected, actual, case):
        seen.append((case["regime"], case["label"]))
        return False, {"compared": True}

    row = candidate(directory)
    assert row["state"] == "NUMERICAL_MISMATCH" and row["metrics"] == {"compared": True}
    assert seen == [key]


def test_a_row_without_check_output_keeps_its_timing_unchecked(tmp_path):
    directory, key = artifacts(tmp_path)
    change(directory, "candidate-name", lambda artifact: artifact["outputs"].pop(key))
    path = directory / "candidate-name" / "result.json"
    result = json.loads(path.read_text())
    result["cells"][0].update(check_unsupported="no read on this system", median_latency_ms=5.0)
    contract.write_json(path, result)
    row = candidate(directory)
    assert row["state"] == "RAN" and row["validation_state"] == "UNRESOLVED"
    assert row["performance_eligible"] and row["median_latency_ms"] == 5.0


def test_a_timed_result_that_disagrees_with_its_check_output_is_invalid(tmp_path):
    directory, _ = artifacts(tmp_path)
    path = directory / "candidate-name" / "result.json"
    result = json.loads(path.read_text())
    result["cells"][0].update(timed_check={"passed": False, "tokens_matched": 0.0},
                              median_latency_ms=5.0)
    contract.write_json(path, result)
    row = candidate(directory)
    assert row["state"] == "INVALID_OUTPUT" and not row["performance_eligible"]
    assert row["validation_state"] == "OUTPUT_CONTRACT_FAILED" and row["median_latency_ms"] == 5.0


def test_reference_failure_does_not_discard_candidate(tmp_path):
    directory, _ = artifacts(tmp_path)
    contract.write_json(directory / "reference-name" / "execution.json", {"status": "failed"})
    assert candidate(directory)["state"] == "NO_REFERENCE"


@pytest.mark.parametrize("field", ["repo", "revision", "vocab_sha256"])
def test_incompatible_identity_refused(tmp_path, field):
    directory, _ = artifacts(tmp_path)
    change(directory, "candidate-name", lambda a: a["provenance"]["model_identity"].update({field: "other"}))
    assert candidate(directory)["state"] == "INCOMPATIBLE"


def test_tampered_input_refused(tmp_path):
    directory, _ = artifacts(tmp_path)
    (directory / "inputs.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        candidate(directory)


def test_batched_correctness_uses_perprompt_reference(tmp_path):
    directory, key = artifacts(tmp_path, batched=True)
    change(directory, "reference-name", lambda a: a["outputs"].update({key: torch.zeros(1, 8)}))
    assert candidate(directory)["state"] == "SUPPORTED"
    assert candidate(directory, comparison="equivalence")["state"] == "DIVERGENT"


@pytest.mark.parametrize("value", [torch.full((1, 8), float("nan")), torch.zeros(2, 8), torch.zeros(1, 4),
                                   torch.tensor([[10., 0., 0., 0., 0., 0., 0., 0., 0.]])])
def test_nonfinite_or_wrong_shape_is_not_rescued(tmp_path, value):
    directory, key = artifacts(tmp_path)
    change(directory, "candidate-name", lambda a: a["outputs"].update({key: value}))
    assert candidate(directory)["state"] == "INVALID_OUTPUT"


def test_missing_tensor_file_is_job_failure(tmp_path):
    directory, _ = artifacts(tmp_path)
    (directory / "candidate-name" / "result.pt").unlink()
    assert candidate(directory)["state"] == "JOB_FAILED"


def test_numerical_mismatch_is_unresolved_and_eligible(tmp_path):
    directory, key = artifacts(tmp_path)
    change(directory, "candidate-name", lambda a: a["outputs"].update({key: torch.zeros(1, 8)}))
    row = candidate(directory)
    assert row["state"] == "NUMERICAL_MISMATCH"
    assert row["validation_state"] == "UNRESOLVED"
    assert row["performance_eligible"] is True
    assert row["metrics"]["tv"] > 0.05


def test_nonfinite_reference_is_not_candidate_failure(tmp_path):
    directory, key = artifacts(tmp_path)
    change(directory, "reference-name", lambda a: a["outputs"].update({key: torch.full((1, 8), float("nan"))}))
    assert candidate(directory)["state"] == "INVALID_REFERENCE"


def test_malformed_reference_keeps_candidate_eligible(tmp_path):
    directory, key = artifacts(tmp_path)
    change(directory, "reference-name", lambda a: a["outputs"].update({key: "bad reference"}))
    row = candidate(directory)
    assert row["state"] == "INVALID_REFERENCE"
    assert row["performance_eligible"]

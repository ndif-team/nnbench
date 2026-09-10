"""Standalone scoring verifies the saved procedure before producing verdicts."""
from dataclasses import replace

import pytest
import torch

from isb.runfile import save_run
from isb.runs import spec_coordinates
from isb.sweep.score import score_runs
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime


def _spec():
    return CellConfig("fixture", "fixture", "gpt2", "repo",
                      [ExecutionRegime("interactive", ["one"])], [({}, "task")], BaselineSpec({}),
                      protocol_absence_reason="scoring identity fixture")


def _save(path, name, spec, *, historical=False):
    coords = spec_coordinates(spec, "hf" if name == "ref" else "vllm_async")
    if historical:
        coords["schema"] = 2
        for key in ("inputs_sha256", "config", "identity_complete"):
            coords.pop(key)
    meta = {("interactive", "task"): {"error": None},
            ("__baseline__", "interactive"): {"error": None},
            ("__effect__", "interactive"): {"strong": True},
            ("batched_perprompt", "task"): {"params": {}}}
    save_run(str(path), name, {("interactive", "task"): torch.tensor([[10., 0.]]),
                               ("__meta__",): meta},
             {"coordinates": coords, "engine": {"kind": "transformers" if name == "ref" else "vllm"}})


def test_matching_complete_procedures_score_only_requested_tasks(tmp_path):
    spec = _spec()
    _save(tmp_path, "ref", spec)
    _save(tmp_path, "cand", spec)
    rows = score_runs(spec, str(tmp_path), "cand", "ref", quiet=True)
    assert len(rows) == 1 and rows[0].label == "task" and rows[0].state == "SUPPORTED"


@pytest.mark.parametrize("historical", ["cand", "ref", "both"])
def test_missing_historical_identity_cannot_produce_a_verdict(tmp_path, historical):
    spec = _spec()
    _save(tmp_path, "ref", spec, historical=historical in ("ref", "both"))
    _save(tmp_path, "cand", spec, historical=historical in ("cand", "both"))
    with pytest.raises(ValueError, match="incomplete"):
        score_runs(spec, str(tmp_path), "cand", "ref", quiet=True)
    rows = score_runs(spec, str(tmp_path), "cand", None, quiet=True)
    assert len(rows) == 1 and rows[0].state == "RAN"


def test_changed_current_spec_is_rejected(tmp_path):
    spec = _spec()
    _save(tmp_path, "ref", spec)
    _save(tmp_path, "cand", spec)
    with pytest.raises(ValueError, match="current spec differs"):
        score_runs(replace(spec, warmup=spec.warmup + 1), str(tmp_path), "cand", "ref", quiet=True)


@pytest.mark.parametrize("target", ["ref", "ctl"])
def test_reference_and_precision_control_must_match_candidate_procedure(tmp_path, target):
    spec = _spec()
    _save(tmp_path, "cand", spec)
    _save(tmp_path, "ref", replace(spec, n_trials=2) if target == "ref" else spec)
    _save(tmp_path, "ctl", replace(spec, n_trials=2) if target == "ctl" else spec)
    with pytest.raises(ValueError, match="procedure is incomplete or different"):
        score_runs(spec, str(tmp_path), "cand", "ref", ctl="ctl", quiet=True)

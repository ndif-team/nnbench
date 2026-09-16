import pytest

from isb.validation import performance_eligible


@pytest.mark.parametrize("state", [
    "RAN", "SUPPORTED", "NUMERICAL_MISMATCH", "DIVERGENT", "NO_REFERENCE",
    "INVALID_REFERENCE", "INCOMPATIBLE", "SILENTLY_WRONG",
])
def test_uncertainty_does_not_exclude_measurement(state):
    assert performance_eligible(state)


@pytest.mark.parametrize("state", ["INVALID_OUTPUT", "CONFIRMED_INCORRECT", "ERROR", "JOB_FAILED", "HANG"])
def test_failed_execution_is_diagnostic_only(state):
    assert not performance_eligible(state)


@pytest.mark.parametrize("state", ["NUMERICAL_MISMATCH", "DIVERGENT", "NO_REFERENCE", "INVALID_REFERENCE", "INCOMPATIBLE"])
def test_strict_cli_keeps_numerical_uncertainty_nonblocking(monkeypatch, tmp_path, state):
    from isb.jobs import cli, score
    monkeypatch.setattr(score, "score_run", lambda _: {"experiments": [{"cells": [{"state": state}]}]})
    assert cli.main(["score", str(tmp_path), "--strict"]) == 0


def test_strict_cli_rejects_invalid_output(monkeypatch, tmp_path):
    from isb.jobs import cli, score
    monkeypatch.setattr(score, "score_run", lambda _: {"experiments": [{"cells": [{"state": "INVALID_OUTPUT"}]}]})
    assert cli.main(["score", str(tmp_path), "--strict"]) == 1

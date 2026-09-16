"""The documented file contract is usable without importing nnbench worker helpers."""
import os
import subprocess
import sys
from pathlib import Path

from isb.jobs.contract import prepare, validate_result
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime


def _job(tmp_path):
    spec = CellConfig("independent", "independent", "fixture", "fixture",
                      [ExecutionRegime("interactive", ["hello"])], [({}, "case")], BaselineSpec({}),
                      protocol_absence_reason="Independent file-contract fixture")
    return prepare(spec, tmp_path / "job")


def _worker(tmp_path, mode):
    fixture = Path(__file__).parent / "fixtures/job_worker.py"
    env = {**os.environ, "ISB_BACKEND": "third-party", "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(fixture), mode, str(tmp_path / "job"),
                           str(tmp_path / "output")], cwd=tmp_path, env=env,
                          text=True, capture_output=True, timeout=45)


def test_independent_v2_worker_is_accepted(tmp_path):
    experiment = _job(tmp_path)
    process = _worker(tmp_path, "success")
    assert process.returncode == 0, process.stderr
    result = validate_result(tmp_path / "output", experiment, "third-party")
    assert result["cells"] == [{"workload": "interactive", "label": "case", "state": "RAN"}]


def test_pinned_v1_worker_rejects_v2_clearly(tmp_path):
    _job(tmp_path)
    process = _worker(tmp_path, "v1-only")
    assert process.returncode != 0
    assert "unsupported experiment version 2; supported: (1,)" in process.stderr
    assert not (tmp_path / "output/result.json").exists()

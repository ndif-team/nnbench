"""Opt-in lifecycle integration: ISB_DOCKER_TESTS=1 pytest tests/test_jobs_docker.py.

Uses an already built HF image for a CPU-only third-party worker, no model or GPU.
Temporary backend directories prove discovery does not require modifying runner code.
"""
import os
from pathlib import Path

import pytest

from isb.jobs import contract, local
from isb.sweep.spec import BaselineSpec, CellConfig, Workload

pytestmark = pytest.mark.skipif(os.environ.get("ISB_DOCKER_TESTS") != "1",
                                reason="opt-in: requires Docker and isb-nnsight-hf:local")


@pytest.mark.parametrize("mode,status", [("success", "completed"), ("crash", "failed"),
                                        ("timeout", "failed")])
def test_third_party_directory_with_real_docker(tmp_path, mode, status):
    package = tmp_path / "backends" / "third-party"
    package.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures" / "job_worker.py"
    # These are runtime test fixtures, not another registered benchmark backend.
    (package / "compose.yml").write_text(f"""services:
  runner:
    image: isb-nnsight-hf:local
    user: "${{ISB_UID}}:${{ISB_GID}}"
    entrypoint: [python3, /worker.py, {mode}]
    environment:
      PYTHONPATH: /workspace
      PYTHONDONTWRITEBYTECODE: "1"
      ISB_BACKEND: "${{ISB_BACKEND}}"
    volumes:
      - {local.ROOT}:/workspace:ro
      - {fixture}:/worker.py:ro
      - ${{ISB_JOB_DIR}}:/job:ro
      - ${{ISB_OUTPUT_DIR}}:/output
""")
    spec = CellConfig("fixture", "fixture", "fixture", "fixture",
                      [Workload("interactive", ["a"])], [({}, "task")], BaselineSpec({}),
                      protocol_absence_reason="Independent Docker lifecycle fixture")
    job = tmp_path / "job"
    experiment = contract.prepare(spec, job)
    assert local.discover(tmp_path) == ["third-party"]
    record = local.run_job("third-party", job, experiment, tmp_path / "output", root=tmp_path,
                           timeout=2 if mode == "timeout" else 60)
    assert record["status"] == status
    assert record["image_id"].startswith("sha256:")
    assert "cleanup_error" not in record
    import subprocess
    remaining = subprocess.run(["docker", "ps", "-aq", "--filter",
                                 f"label=com.docker.compose.project={record['project']}"],
                                capture_output=True, text=True, check=True)
    assert not remaining.stdout.strip()

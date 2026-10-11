"""Apptainer launch path: Dockerfile translation, compose mapping and a job lifecycle. No apptainer,
Docker or GPU needed."""
import subprocess

import pytest

from isb.jobs import apptainer, contract, local
from tests.test_jobs import completed, specimen

DOCKERFILE = """\
ARG ENGINE_IMAGE=example/engine:v1
FROM ${ENGINE_IMAGE}
# tooling comment between instructions
ARG PROBE_VERSION=0.3
RUN apt-get install -y git \\
      && pip install "probe-kit==${PROBE_VERSION}" \\
      && echo $PATH
RUN pip check
"""


def test_dockerfile_translates_args_from_and_runs(tmp_path):
    (tmp_path / "Dockerfile").write_text(DOCKERFILE)
    recipe = apptainer.definition(tmp_path / "Dockerfile", {"PROBE_VERSION": "0.9"})
    assert recipe.startswith("Bootstrap: docker\nFrom: example/engine:v1\n")
    assert '"probe-kit==0.9"' in recipe                     # a build arg overrides the default
    assert "echo $PATH" in recipe                           # unknown variables stay for the shell
    assert "\\\n" in recipe and "pip check" in recipe and "tooling comment" not in recipe


@pytest.mark.parametrize("text,error", [
    ("FROM a\nCOPY x /x\n", "COPY is not translated"),
    ("FROM a\nFROM b\n", "multi-stage"),
    ("ARG BASE\nFROM ${BASE}\n", "ARG BASE has no value"),
])
def test_untranslatable_dockerfiles_are_refused(tmp_path, text, error):
    (tmp_path / "Dockerfile").write_text(text)
    with pytest.raises(ValueError, match=error):
        apptainer.definition(tmp_path / "Dockerfile", {})


COMPOSE = """\
services:
  runner:
    image: probe-system:local
    working_dir: /workspace
    entrypoint: [python3, /backend/run.py]
    environment:
      ISB_BACKEND: ${ISB_BACKEND:-probe-system/default}
      ISB_ENGINE_OPTIONS: '{"dtype":"bfloat16","gpu_memory_utilization":0.2}'
      HF_TOKEN: ${HF_TOKEN:-}
    volumes:
      - ../../..:/workspace:ro
      - ${ISB_OUTPUT_DIR:-./out}:/output
      - weights:/models
volumes:
  weights:
    name: ${ISB_MODEL_CACHE:-probe-weights}
"""


def test_compose_maps_to_apptainer_exec(tmp_path, monkeypatch):
    monkeypatch.setenv("ISB_APPTAINER_DIR", str(tmp_path / "store"))
    backend = tmp_path / "repo" / "backends" / "probe-system" / "default"
    backend.mkdir(parents=True)
    (backend / "compose.yml").write_text(COMPOSE)
    env = {"ISB_BACKEND": "probe-system/default", "ISB_GPU": "3", "ISB_OUTPUT_DIR": str(tmp_path / "o")}
    with pytest.raises(ValueError, match="missing"):
        apptainer.command(backend / "compose.yml", tmp_path / "repo", env)
    image = apptainer.sif_path(tmp_path / "repo", "probe-system:local")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"sif")
    argv, extra = apptainer.command(backend / "compose.yml", tmp_path / "repo", env)
    assert argv[:5] == ["apptainer", "exec", "--nv", "--cleanenv", "--no-home"]
    assert argv[-3:] == [str(image), "python3", "/backend/run.py"]
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert f"{tmp_path / 'repo'}:/workspace:ro" in binds
    assert f"{tmp_path / 'o'}:/output" in binds
    assert f"{tmp_path / 'store' / 'volumes' / 'probe-weights'}:/models" in binds
    assert extra["APPTAINERENV_ISB_ENGINE_OPTIONS"] == '{"dtype":"bfloat16","gpu_memory_utilization":0.2}'
    assert extra["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "3"
    assert extra["APPTAINERENV_HF_TOKEN"] == ""


def test_supporting_services_need_docker(tmp_path):
    (tmp_path / "compose.yml").write_text(
        "services:\n  runner: {image: a, entrypoint: [x]}\n  cache: {image: b}\n")
    with pytest.raises(ValueError, match="supporting services"):
        apptainer.load_service(tmp_path / "compose.yml", {})


def test_job_lifecycle_through_apptainer(tmp_path, monkeypatch):
    monkeypatch.setenv("ISB_APPTAINER_DIR", str(tmp_path / "store"))
    backend = tmp_path / "backends" / "probe-system" / "default"
    backend.mkdir(parents=True)
    (backend / "compose.yml").write_text(COMPOSE)
    image = apptainer.sif_path(tmp_path, "probe-system:local")
    image.parent.mkdir(parents=True)
    image.write_bytes(b"sif image bytes")
    experiment = contract.prepare(specimen(), tmp_path / "job")
    monkeypatch.setattr(local, "gpu_used_mib", lambda gpu: None)
    launched = []

    def fake_run(argv, extra_env, log, timeout):
        launched.append(argv)
        completed(tmp_path / "output", experiment, "probe-system/default")
        return 0

    monkeypatch.setattr(apptainer, "run", fake_run)
    real_run = subprocess.run

    def no_docker(argv, *args, **kwargs):
        assert argv[0] != "docker", f"Docker was called: {argv}"
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(local.subprocess, "run", no_docker)
    record = local.run_job("probe-system/default", tmp_path / "job", experiment, tmp_path / "output",
                           root=tmp_path, launcher="apptainer")
    assert record["status"] == "completed" and record["launcher"] == "apptainer"
    assert record["image_id"] == "sha256:" + contract.file_digest(image)
    assert launched[0][0:2] == ["apptainer", "exec"]


def test_timeout_kills_the_whole_process_group(tmp_path):
    with (tmp_path / "log").open("w") as log, pytest.raises(subprocess.TimeoutExpired):
        apptainer.run(["sh", "-c", "sleep 30 & sleep 30"], {}, log, timeout=0.5)

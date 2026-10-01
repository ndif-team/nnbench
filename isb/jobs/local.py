"""Local Docker Compose job lifecycle. Backend names are directory names, never engine keys."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from . import apptainer
from .contract import digest, file_digest, validate_result, write_json

LAUNCHERS = ("docker", "apptainer")


def default_launcher():
    """The runner's launch method: ISB_LAUNCHER, else Docker. A host setting, never per backend."""
    launcher = os.environ.get("ISB_LAUNCHER", "docker")
    if launcher not in LAUNCHERS:
        raise ValueError(f"ISB_LAUNCHER must be one of {LAUNCHERS}, not {launcher!r}")
    return launcher

ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")


def discover(root=ROOT):
    return sorted(p.name for p in (root / "backends").iterdir()
                  if NAME.fullmatch(p.name) and (p / "compose.yml").is_file()
                  and p.resolve().parent == (root / "backends").resolve())


def backend_file(name, root=ROOT):
    if not NAME.fullmatch(name) or name not in discover(root):
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(discover(root))}")
    path = root / "backends" / name / "compose.yml"
    if path.resolve().parent != (root / "backends" / name).resolve():
        raise ValueError("backend configuration must be inside its directory")
    return path


def compose(name, project, root=ROOT):
    return ["docker", "compose", "--project-name", project, "--file", str(backend_file(name, root))]


def environment(name, job, output, gpu):
    return {**os.environ, "ISB_BACKEND": name, "ISB_JOB_DIR": str(job.resolve()),
            "ISB_OUTPUT_DIR": str(output.resolve()), "ISB_GPU": str(gpu),
            "ISB_UID": str(os.getuid()), "ISB_GID": str(os.getgid())}


def configuration(name, root=ROOT, env=None):
    result = subprocess.run(compose(name, "isb-validate", root) + ["config", "--format", "json"],
                            env=env, capture_output=True, text=True, check=True, timeout=30)
    config = json.loads(result.stdout)
    service = config.get("services", {}).get("runner")
    if service is None:
        raise ValueError(f"{name}: compose.yml must define a runner service")
    # Fixed container names/external networks undermine per-job project isolation.
    for item in config["services"].values():
        if item.get("container_name") or item.get("network_mode") == "host":
            raise ValueError(f"{name}: fixed container_name and host networking are not isolated")
        if item.get("ports"):
            raise ValueError(f"{name}: use the private Compose network, not published host ports")
    if any(n.get("external") for n in config.get("networks", {}).values()):
        raise ValueError(f"{name}: external networks are not job-isolated")
    return config


def source_identity(root=ROOT):
    files = sorted(p for folder in ("isb", "backends") for p in (root / folder).rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts
                   and (p.suffix in {".py", ".yml", ".yaml"} or p.name == "Dockerfile"))
    content = "\n".join(f"{p.relative_to(root)} {file_digest(p)}" for p in files)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
    return {"commit": commit.stdout.strip(), "source_sha256": digest(content.encode())}


def gpu_used_mib(gpu):
    """Used memory on the selected GPU in MiB; None when nvidia-smi cannot report it."""
    try:
        result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return int(value) if result.returncode == 0 and value.isdigit() else None


def wait_for_release(gpu, baseline, *, tolerance_mib=1024, timeout=120, poll=1.0):
    """Wait until the GPU's used memory is back within `tolerance_mib` of `baseline`, at most
    `timeout` seconds. A finished job's GPU memory can still be draining after `compose down`
    returns; a vLLM engine that starts meanwhile fails its memory profiling. Returns
    (used_mib, waited_seconds); used_mib is None when the GPU cannot be read."""
    start = time.monotonic()
    used = gpu_used_mib(gpu)
    while (baseline is not None and used is not None and used > baseline + tolerance_mib
           and time.monotonic() - start < timeout):
        time.sleep(poll)
        used = gpu_used_mib(gpu)
    return used, round(time.monotonic() - start, 1)


def run_job(name, job, experiment, output, *, gpu="0", timeout=1800, root=ROOT, launcher="docker"):
    """Always collect an execution record and clean only this job's Compose project (Docker) or
    process group (Apptainer)."""
    output.mkdir(parents=True, exist_ok=False)
    project = "isb-" + uuid.uuid4().hex[:20]
    container = project + "-runner"
    command = compose(name, project, root)
    env = environment(name, job, output, gpu)
    record = {"backend": name, "experiment_id": experiment["id"], "project": project,
              "launcher": launcher, "status": "running", "source": source_identity(root)}
    write_json(output / "execution.json", record)
    baseline_mib = gpu_used_mib(gpu)
    print(f"[run] {name} / {experiment['spec']['name']} → {output}", flush=True)
    try:
        if launcher == "apptainer":
            argv, extra_env = apptainer.command(backend_file(name, root), root, env)
            with (output / "execution.log").open("w") as log:
                exit_code = apptainer.run(argv, extra_env, log, timeout)
        else:
            configuration(name, root, env)
            with (output / "execution.log").open("w") as log:
                exit_code = subprocess.run(command + ["run", "--name", container, "--no-TTY", "runner"],
                                           env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=timeout).returncode
        record["exit_code"] = exit_code
        if exit_code:
            raise RuntimeError(f"container exited {exit_code}; see execution.log")
        validate_result(output, experiment, name)
        if record["source"] != source_identity(root):
            raise RuntimeError("benchmark/backend source changed during job; rerun with stable sources")
        record["status"] = "completed"
    except subprocess.TimeoutExpired:
        record.update(status="failed", error=f"job exceeded {timeout} seconds")
    except KeyboardInterrupt:
        record.update(status="cancelled", error="interrupted")
        raise
    except Exception as error:
        record.update(status="failed", error=str(error))
    finally:
        try:
            if launcher == "apptainer":
                # The process group is gone (run() waited or killed it); the image is the .sif.
                service, _ = apptainer.load_service(backend_file(name, root), env)
                record["image_id"] = apptainer.image_identity(apptainer.sif_path(root, service["image"]))
            else:
                # Inspect the container's actual image, not a tag that may have moved mid-run.
                inspected = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", container],
                                           capture_output=True, text=True, timeout=15)
                if inspected.returncode == 0:
                    record["image_id"] = inspected.stdout.strip()
                elif record["status"] == "completed":
                    record.update(status="failed", error="could not record executed image identity")
                with (output / "cleanup.log").open("w") as log:
                    cleaned = subprocess.run(command + ["down", "--remove-orphans", "--timeout", "10"],
                                             env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
                if cleaned.returncode:
                    record.update(status="failed", cleanup_error="Compose cleanup failed; see cleanup.log")
            after_mib, waited = wait_for_release(gpu, baseline_mib)
            record["gpu_release"] = {"baseline_mib": baseline_mib, "after_mib": after_mib,
                                     "waited_s": waited}
        except Exception as error:
            record.update(status="failed", cleanup_error=str(error))
        write_json(output / "execution.json", record)
        print(f"[{record['status']}] {name}: {record.get('error', '')}", flush=True)
    return record

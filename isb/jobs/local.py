"""Local Docker Compose job lifecycle. Backend names are directory names, never engine keys."""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from .contract import digest, file_digest, validate_result, write_json

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


def run_job(name, job, experiment, output, *, gpu="0", timeout=1800, root=ROOT):
    """Always collect an execution record and clean only this job's Compose project."""
    output.mkdir(parents=True, exist_ok=False)
    project = "isb-" + uuid.uuid4().hex[:20]
    container = project + "-runner"
    command = compose(name, project, root)
    env = environment(name, job, output, gpu)
    record = {"backend": name, "experiment_id": experiment["id"], "project": project,
              "status": "running", "source": source_identity(root)}
    write_json(output / "execution.json", record)
    print(f"[run] {name} / {experiment['spec']['name']} → {output}", flush=True)
    try:
        configuration(name, root, env)
        with (output / "execution.log").open("w") as log:
            result = subprocess.run(command + ["run", "--name", container, "--no-TTY", "runner"],
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        record["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"container exited {result.returncode}; see execution.log")
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
        except Exception as error:
            record.update(status="failed", cleanup_error=str(error))
        write_json(output / "execution.json", record)
        print(f"[{record['status']}] {name}: {record.get('error', '')}", flush=True)
    return record

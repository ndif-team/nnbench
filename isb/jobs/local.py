"""Local Docker Compose job lifecycle. Backend names are catalog entries, `<system>/<config>`
(isb/jobs/layout.py), never engine keys."""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

from . import apptainer
from .contract import digest, file_digest, validate_result, write_json
from .layout import ROOT, canonical, config, discover, system_dir  # noqa: F401  (re-exported for the CLI)

LAUNCHERS = ("docker", "apptainer")


def default_launcher():
    """The runner's launch method: ISB_LAUNCHER, else Docker. A host setting, never per backend."""
    launcher = os.environ.get("ISB_LAUNCHER", "docker")
    if launcher not in LAUNCHERS:
        raise ValueError(f"ISB_LAUNCHER must be one of {LAUNCHERS}, not {launcher!r}")
    return launcher


def compose_file(name, root=ROOT):
    """compose.yml of the backend's system."""
    return system_dir(canonical(name, root), root) / "compose.yml"


def compose(name, project, root=ROOT):
    return ["docker", "compose", "--project-name", project, "--file", str(compose_file(name, root))]


def gpu_list(gpu, count):
    """The host GPUs a config reserves: `--gpu` and, for a config reserving several, the indices
    after it."""
    if count == 1:
        return [str(gpu)]
    if not str(gpu).isdigit():
        raise ValueError(f"a config reserving {count} GPUs needs --gpu to be an index")
    return [str(int(gpu) + k) for k in range(count)]


def environment(name, job, output, gpu, root=ROOT):
    """The job's Compose environment: the config's image, options, GPUs and ranks
    (backends/README.md)."""
    chosen = config(name, root)
    return {**os.environ, "ISB_BACKEND": chosen.name, "ISB_IMAGE": chosen.tag,
            "ISB_ENGINE_OPTIONS": json.dumps(chosen.options, separators=(",", ":")),
            "ISB_GPUS": ",".join(gpu_list(gpu, chosen.gpus)), "ISB_RANKS": str(chosen.ranks),
            "ISB_JOB_DIR": str(job.resolve()), "ISB_OUTPUT_DIR": str(output.resolve()),
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


IDLE_MIB = 100     # an idle GPU holds only driver residue, tens of MiB


CLOCK_FIELDS = ("clocks.sm", "clocks.max.sm", "clocks.applications.graphics", "clocks.mem",
                "enforced.power.limit", "power.draw", "temperature.gpu", "persistence_mode",
                "clocks_event_reasons.active")


def _smi(gpus, query, fields=4):
    result = subprocess.run(["nvidia-smi", "-i", ",".join(gpus), query, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise OSError(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
    return [[field.strip() for field in line.split(",", fields - 1)]
            for line in result.stdout.strip().splitlines()]


def gpu_clocks(gpus):
    """Each GPU's clocks, enforced power limit, power draw, temperature and active clock event
    reasons as nvidia-smi reports them, keyed by index; None when nvidia-smi cannot report them."""
    try:
        rows = _smi(gpus, "--query-gpu=index," + ",".join(CLOCK_FIELDS), len(CLOCK_FIELDS) + 1)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {row[0]: dict(zip(CLOCK_FIELDS, row[1:])) for row in rows}


def gpu_state(gpus):
    """Used memory (MiB) and compute processes of each GPU in `gpus`, keyed by the index
    nvidia-smi reports; None when nvidia-smi cannot report them."""
    try:
        memory = _smi(gpus, "--query-gpu=index,pci.bus_id,memory.used")
        apps = _smi(gpus, "--query-compute-apps=gpu_bus_id,pid,used_memory,process_name")
    except (OSError, subprocess.TimeoutExpired):
        return None
    state = {index: {"used_mib": int(used), "processes": []} for index, _, used in memory}
    bus = {bus_id: index for index, bus_id, _ in memory}
    for bus_id, pid, used, name in apps:
        state[bus[bus_id]]["processes"].append(
            {"pid": int(pid), "used_mib": int(used) if used.isdigit() else None, "name": name})
    return state


def busy(state):
    """The GPUs of `state` that hold another process's memory or compute context."""
    return {index: gpu for index, gpu in state.items()
            if gpu["used_mib"] >= IDLE_MIB or gpu["processes"]}


def wait_idle(gpus, *, timeout=120, poll=1.0):
    """Wait until no GPU in `gpus` is busy, at most `timeout` seconds: a finished job's memory can
    still be draining after its container exits. Returns (state, waited_seconds); state is None
    when nvidia-smi cannot report."""
    start = time.monotonic()
    state = gpu_state(gpus)
    while state is not None and busy(state) and time.monotonic() - start < timeout:
        time.sleep(poll)
        state = gpu_state(gpus)
    return state, round(time.monotonic() - start, 1)


def refuse_busy(state):
    """Raise when a reserved GPU is busy: another process's kernels and memory traffic would sit
    inside every timed call, and the job would disturb that process."""
    occupied = busy(state or {})
    if occupied:
        raise RuntimeError("reserved GPU in use before the job: " + "; ".join(
            f"GPU {index}: {gpu['used_mib']} MiB used"
            + "".join(f", pid {p['pid']} {p['name']} ({p['used_mib']} MiB)" for p in gpu["processes"])
            for index, gpu in occupied.items()))


def run_job(name, job, experiment, output, *, gpu="0", timeout=1800, root=ROOT, launcher="docker"):
    """Always collect an execution record and clean only this job's Compose project (Docker) or
    process group (Apptainer)."""
    # A config's job directory may sit inside another's (0-8-1/opt in 0-8-1): the directory may
    # exist, a previous job's record in it may not.
    output.mkdir(parents=True, exist_ok=True)
    if (output / "execution.json").exists():
        raise FileExistsError(f"{output} already holds a job")
    project = "isb-" + uuid.uuid4().hex[:20]
    container = project + "-runner"
    command = compose(name, project, root)
    env = environment(name, job, output, gpu, root)
    record = {"backend": name, "experiment_id": experiment["id"], "project": project,
              "launcher": launcher, "status": "running", "source": source_identity(root)}
    write_json(output / "execution.json", record)
    print(f"[run] {name} / {experiment['spec']['name']} → {output}", flush=True)
    try:
        state, waited = wait_idle(env["ISB_GPUS"].split(","))
        record["gpus_before"] = {"state": state, "waited_s": waited,
                                 "clocks": gpu_clocks(env["ISB_GPUS"].split(","))}
        refuse_busy(state)
        if launcher == "apptainer":
            argv, extra_env = apptainer.command(compose_file(name, root), root, env)
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
        record["gpus_after"] = {"clocks": gpu_clocks(env["ISB_GPUS"].split(","))}
        try:
            if launcher == "apptainer":
                # The process group is gone (run() waited or killed it); the image is the .sif.
                service, _ = apptainer.load_service(compose_file(name, root), env)
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
        except Exception as error:
            record.update(status="failed", cleanup_error=str(error))
        write_json(output / "execution.json", record)
        print(f"[{record['status']}] {name}: {record.get('error', '')}", flush=True)
    return record

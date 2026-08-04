"""Release-mode environment preflight: assert the conditions under which a measurement is valid.

Perf numbers taken on a contended GPU are invalid even when the run completes: another job's
kernels and memory traffic sit inside every timed trial. So `bench.py --release` checks the
environment BEFORE any model load and REFUSES on contamination, naming the offending processes.
It never waits or schedules (cluster tooling, not benchmark logic), and outside release mode it
does not run at all: a casual run just runs.

The findings are returned as data so release-mode provenance can attach the clean-environment
evidence to the numbers it certifies.
"""
from __future__ import annotations

import os
import shutil
import subprocess

# a GPU is "occupied" above this floor: allows the ~dozens of MB of driver/display residue
_OCCUPIED_MB = 512
# below this free space, output/refs writes are at risk (this week's ENOSPC mid-write)
_MIN_FREE_DISK_GB = 5


def _visible_indices() -> list[int] | None:
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v is None:
        return None                              # all GPUs visible
    return [int(x) for x in v.split(",") if x.strip() != ""]


def parse_compute_apps(csv_text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name`-style CSV
    (index queried separately below; kept pure for tests)."""
    rows = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu": parts[0], "pid": parts[1],
                         "used_mb": parts[2].split()[0], "name": parts[3]})
    return rows


def check_environment(out_dir: str = ".") -> list[str]:
    """The release-mode gate. Returns a list of human-readable findings; empty = clean."""
    findings: list[str] = []

    # GPU occupancy on the visible devices
    try:
        mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        visible = _visible_indices()
        if mem.returncode == 0:
            for line in mem.stdout.strip().splitlines():
                idx, used = [x.strip() for x in line.split(",")]
                if visible is not None and int(idx) not in visible:
                    continue
                if int(used) > _OCCUPIED_MB:
                    findings.append(f"GPU {idx}: {used} MiB already in use by other work")
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory,process_name",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if apps.returncode == 0 and apps.stdout.strip():
            for row in parse_compute_apps(apps.stdout):
                findings.append(
                    f"compute process on {row['gpu']}: pid {row['pid']} "
                    f"({row['name']}, {row['used_mb']} MiB)")
    except OSError:
        findings.append("nvidia-smi unavailable: GPU occupancy cannot be verified")

    # disk headroom where outputs/refs land
    free_gb = shutil.disk_usage(out_dir).free / 2**30
    if free_gb < _MIN_FREE_DISK_GB:
        findings.append(f"only {free_gb:.1f} GB free at {out_dir!r} "
                        f"(< {_MIN_FREE_DISK_GB} GB): output writes at risk")
    return findings

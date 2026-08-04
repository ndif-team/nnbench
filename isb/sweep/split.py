"""Split-run orchestration (design.md §12.9): every run is its own scripts/execute.py process.

Co-residency of the control and the system under test in one process caused real failures:
the object-patch import-order crash (HF traces first, then vLLM's import chain hits a
library whose classes can no longer be defined), the transformers version conflict (the
control needs an arch newer than the vLLM env's pin, e.g. qwen3_5), shared-GPU teardown
ordering, and CUDA-init in the parent forcing vLLM's worker spawn mode. The handoff between
the processes is the run file itself (isb/runfile.py): each subprocess writes ONE
self-contained run (outputs + provenance) into the shared out dir, and scoring is a separate
pure-CPU step over those files (isb/sweep/score.py) — the same contract hand-run
execute/score and the manager already use. A per-backend interpreter map lets each backend
run in its own conda env, which is what makes newer-than-the-vllm-pin models benchable at all.
"""
from __future__ import annotations

import sys

# backend name (the cell-registry vocabulary) -> (run-name tag, execute.py engine coordinates)
ENGINE_ARGS = {
    "hf": ("hf", ["--engine", "transformers"]),
    "vllm_async": ("vllm", ["--engine", "vllm"]),
    "vllm_sync": ("vllm-sync", ["--engine", "vllm", "--engine-mode", "sync"]),
    "vllm_serve": ("serve", ["--engine", "vllm", "--deployment", "serve"]),
}
DTYPE_TAG = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


def run_base(spec_name: str, data: str | None) -> str:
    """The shared run-name stem: `<spec>` or `<spec>@<source>` (--data), so runs of the same
    spec on different data never collide in one directory."""
    if not data:
        return spec_name
    return f"{spec_name}@{data.partition(':')[0].replace('/', '-')}"


def ctl_run_name(spec_name: str, data: str | None, ctl_dtype: str) -> str:
    return f"{run_base(spec_name, data)}-vllm-{DTYPE_TAG.get(ctl_dtype, ctl_dtype)}"


def ctl_run_command(script: str, spec_name: str, out_dir: str, ctl_dtype: str, *,
                    python: str | None = None, data: str | None = None,
                    release: bool = False) -> list:
    """The control-dtype vLLM run: the candidates' engine at the spec's control dtype. score.py
    uses its run file (--ctl) to tell a low-precision near-tie (SUPPORTED_DEGRADED) from a real
    mechanism bug."""
    argv = [python or sys.executable, script, "--spec", spec_name, "--engine", "vllm",
            "--param", f"dtype={ctl_dtype}",
            "--name", ctl_run_name(spec_name, data, ctl_dtype), "--out", out_dir]
    return argv + (["--data", data] if data else []) + (["--release"] if release else [])


def backend_run_commands(script, spec_name, backends, out_dir, *, python_map=None,
                         serve=None, data=None, release=False, ctl_dtype=None):
    """The per-backend execute.py argv lists for one spec, as (backend, run_name, argv) rows:
    the reference run first (hf when present, else the first backend), then one run per
    remaining backend, then — when `ctl_dtype` is given and the reference is hf — the
    control-dtype vLLM run (row label "ctl"). The caller scores every later run against the
    first row's run file. `python_map` maps a backend name to an interpreter path (default:
    the current one), so each backend runs in its own env."""
    python_map = python_map or {}
    control = "hf" if "hf" in backends else backends[0]
    base = run_base(spec_name, data)
    common = (["--data", data] if data else []) + (["--release"] if release else [])

    def py(b):
        return python_map.get(b, sys.executable)

    rows = []
    for b in [control] + [b for b in backends if b != control]:
        tag, engine_args = ENGINE_ARGS[b]
        argv = [py(b), script, "--spec", spec_name, *engine_args]
        if b == "vllm_serve":
            if not serve:
                raise ValueError("the vllm_serve backend needs a server URL (--serve)")
            argv += ["--host", serve]
        name = f"{base}-{tag}"
        rows.append((b, name, argv + ["--name", name, "--out", out_dir] + common))
    if ctl_dtype and control == "hf" and len(backends) > 1:
        vllm_backend = next(b for b in backends if b != "hf")
        rows.append(("ctl", ctl_run_name(spec_name, data, ctl_dtype),
                     ctl_run_command(script, spec_name, out_dir, ctl_dtype,
                                     python=python_map.get(vllm_backend),
                                     data=data, release=release)))
    return rows


def parallel_run_commands(script, spec_name, out_dir, *, pp, tp, executor,
                          gpu_mem, max_model_len, dtype_control):
    """The two PP/TP-equivalence runs (design §12.2), each its own process, as (label, run_name,
    argv) rows: the single-GPU (1,1) vLLM control first, then the (tp,pp) candidate the caller
    scores against it (same engine kind in both provenances -> score.py picks the equivalence
    axis). Both pin the spec's control dtype and the same memory knobs so the only variable is
    the engine topology. Same env and stack in both (they are both vLLM), so no interpreter map;
    the isolation is engine-level: one engine per process, the GPU fully released in between,
    and a crash in one cannot lose the other's results."""
    def argv_for(name, extra_params):
        argv = [sys.executable, script, "--spec", spec_name, "--engine", "vllm",
                "--name", name, "--out", out_dir,
                "--param", f"dtype={dtype_control}",
                "--param", f"gpu_memory_utilization={gpu_mem}"]
        if max_model_len is not None:
            argv += ["--param", f"max_model_len={max_model_len}"]
        for kv in extra_params:
            argv += ["--param", kv]
        return argv

    cand_params = [f"tensor_parallel_size={tp}", f"pipeline_parallel_size={pp}"]
    if executor:
        cand_params.append(f"distributed_executor_backend={executor}")
    ctl_name, cand_name = f"{spec_name}-tp1pp1", f"{spec_name}-tp{tp}pp{pp}"
    return [("vllm (1,1) control", ctl_name, argv_for(ctl_name, [])),
            (f"vllm (tp={tp},pp={pp})", cand_name, argv_for(cand_name, cand_params))]

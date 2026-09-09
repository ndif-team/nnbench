"""Run configuration + provenance (design.md §12.10): a run is a fully described execution.

A benchmark result's identity is not a backend name. It is the composition of four layers,
each DECLARED in the config and/or RECORDED (resolved empirically at run start):

  engine      WHAT executes the model: transformers | vllm, its drive mode, its params
              (dtype, tp/pp, max_model_len, prefix caching, ... — the design §6 config axes)
  deployment  WHERE that engine runs: local | serve | ndif — a hosting layer with its OWN code
              identity (host, the deployment's commit, its env). NDIF is a deployment, not an
              engine: transformers or vllm runs INSIDE it.
  client      the interpreter driving the run (conda env -> library versions -> nnsight checkout)
  host        the hardware and system it ran on (GPU model/count/VRAM, driver, CUDA, CPU, RAM)

Nothing here provisions, mutates envs, or refuses to run: declared values are recorded verbatim
next to what actually resolved, so drift is visible in the record. Everything resolved lands in
the run's provenance file, written next to its outputs, so a comparison can NAME what was
compared (the missing piece that turned a branch skew into a two-day hunt: two
"identical" HF runs on different nnsight commits were indistinguishable in every report).
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata


@dataclass(frozen=True)
class EngineConfig:
    kind: str                    # "transformers" | "vllm"
    mode: str = "async"          # vllm local drive style: "async" | "sync"; ignored for transformers
    params: dict = field(default_factory=dict)   # dtype, tensor_parallel_size, max_model_len, ...


@dataclass(frozen=True)
class DeploymentConfig:
    kind: str = "local"          # "local" | "serve" | "ndif"
    host: str | None = None      # serve URL / ndif endpoint
    api_key_env: str | None = None   # NAME of the env var holding the key; never the key itself
    commit: str | None = None    # the deployment's own code identity (recorded; e.g. ndif commit)
    env: str | None = None       # the deployment's environment descriptor (conda env / image tag)


@dataclass(frozen=True)
class RunConfig:
    engine: EngineConfig
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    python: str | None = None    # client interpreter; None = this one
    nnsight: str | None = None   # the client nnsight commit this config was written against
    gpu_model: str | None = None  # the GPU model this config was written against
    # Both are DESCRIPTIVE: recorded verbatim in provenance next to what actually resolved,
    # so drift is visible in the record; nothing refuses to run.

    def interpreter(self) -> str:
        return self.python or sys.executable


# ---- executor mapping: (engine.kind, engine.mode, deployment.kind) -> backend instance ---------

def make_backend(run: RunConfig, spec):
    """The thin executor for a run config. Engine params merge two sources: the SPEC's required
    fragment (what the model needs to load at all: trust_remote_code, force_text_causal) and the
    RUN's chosen fragment (what the experiment varies). A key present in both with different
    values is a loud error — an experiment must not silently override a model requirement."""
    from .backends import HFBackend, VLLMAsyncBackend, VLLMServeBackend, VLLMSyncBackend

    e, d = run.engine, run.deployment
    if e.kind == "transformers":
        required = dict(spec.hf_kwargs)
    else:
        required = dict(spec.vllm_kwargs)
    merged = dict(required)
    for k, v in e.params.items():
        if k in required and required[k] != v:
            raise ValueError(
                f"engine param {k!r}: run config says {v!r} but the spec requires "
                f"{required[k]!r} (a model-load requirement; vary it in a different spec)")
        merged[k] = v

    if d.kind == "ndif":
        raise NotImplementedError(
            "ndif deployment: config slot reserved (design §3.8); executor lands with the "
            "NDIF-plane scope decision")
    if e.kind == "transformers":
        if d.kind != "local":
            raise ValueError(f"transformers engine supports local deployment only (got {d.kind!r})")
        return HFBackend(**merged)
    if e.kind == "vllm":
        if d.kind == "serve":
            if not d.host:
                raise ValueError("serve deployment needs a host URL")
            return VLLMServeBackend(host=d.host, **merged)
        if d.kind == "local":
            return VLLMSyncBackend(**merged) if e.mode == "sync" else VLLMAsyncBackend(**merged)
    raise ValueError(f"no executor for engine={e.kind!r} mode={e.mode!r} deployment={d.kind!r}")


def cell_interface(run: RunConfig) -> str:
    """The registry key this run's cells are looked up under. Cells are keyed by ENGINE INTERFACE
    only — deployment and topology never change intervention code (the serve fallback measured
    that), so all vllm variants resolve through the existing vllm_async fallback chain."""
    if run.engine.kind == "transformers":
        return "hf"
    if run.engine.kind == "vllm":
        if run.deployment.kind == "serve":
            return "vllm_serve"
        return "vllm_sync" if run.engine.mode == "sync" else "vllm_async"
    raise ValueError(f"no cell interface for engine {run.engine.kind!r}")


# ---- provenance: resolve empirically, record everything (descriptive, never a gate) -----------

def _git_identity(path: str) -> dict:
    """Commit + dirty flag of the checkout containing `path`; {} when not a git checkout."""
    try:
        top = subprocess.run(["git", "-C", path, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
        if top.returncode != 0:
            return {}
        commit = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip() != ""
        remote = subprocess.run(["git", "-C", path, "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=10)
        return {"commit": commit, "dirty": dirty, "checkout": top.stdout.strip(),
                "remote": remote.stdout.strip() if remote.returncode == 0 else None}
    except OSError:
        return {}


def _version_of(module_name: str) -> str | None:
    try:
        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def _installed_git_identity(package: str) -> dict:
    """Pip retains the full commit for a Git-installed wheel even without a .git directory."""
    try:
        info = json.loads(metadata.distribution(package).read_text("direct_url.json") or "{}")
        vcs = info.get("vcs_info", {})
        if vcs.get("vcs") == "git" and vcs.get("commit_id"):
            return {"commit": vcs["commit_id"], "remote": info.get("url")}
    except (metadata.PackageNotFoundError, ValueError, OSError):
        pass
    return {}


def _gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        return {
            "available": True,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": [{"name": p.name, "vram_gb": round(p.total_memory / 2**30, 1)} for p in props],
            "cuda": torch.version.cuda,
            "driver": _nvidia_driver(),
        }
    except ImportError:
        return {"available": False}


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()[0].strip() if out.returncode == 0 else None
    except OSError:
        return None


def resolve_provenance(run: RunConfig) -> dict:
    """The four-layer provenance record, resolved IN the run's container (not the orchestrator —
    the whole point is recording the stack that executed).
    Purely DESCRIPTIVE: declared expectations (nnsight commit, gpu model) are recorded alongside
    what was actually resolved, so a mismatch is visible in the record — never a refusal."""
    from importlib.util import find_spec

    module = find_spec("nnsight")
    nnsight_path = os.path.dirname(module.origin) if module and module.origin else None
    nnsight_git = (_git_identity(nnsight_path) if nnsight_path else {}) or _installed_git_identity("nnsight")
    gpu = _gpu_info()

    return {
        "executed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "declared": {  # what the config said, kept verbatim next to what was resolved
            "nnsight": run.nnsight,
            "gpu_model": run.gpu_model,
            "python": run.python,
        },
        "client": {
            "python": sys.executable,
            "nnsight": {"path": nnsight_path, "version": _version_of("nnsight"), **nnsight_git},
            "torch": _version_of("torch"),
            "transformers": _version_of("transformers"),
            "vllm": _version_of("vllm"),
        },
        "deployment": asdict(run.deployment),
        "engine": {"kind": run.engine.kind, "mode": run.engine.mode, "params": run.engine.params},
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            **gpu,
        },
    }

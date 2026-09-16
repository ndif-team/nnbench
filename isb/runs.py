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
import math
import os
import platform
import socket
import subprocess
import sys
import time
from copy import deepcopy
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


# ---- run coordinates: the one shape every run file's `coordinates` take ------------------------

COORDINATES_SCHEMA = 3
_CONFIG_FIELDS = ("baseline", "effect", "dtype_control", "warmup", "n_trials", "hf_kwargs", "vllm_kwargs")
_COMPARISON_CONFIG_FIELDS = ("baseline", "effect", "hf_kwargs", "vllm_kwargs")


def coordinate_config(spec_or_record) -> dict:
    """The spec controls that affect execution, timing, or interpretation of its outputs."""
    record = spec_or_record if isinstance(spec_or_record, dict) else asdict(spec_or_record)
    return deepcopy({key: record[key] for key in _CONFIG_FIELDS if key in record})


def _complete_identity(inputs_sha256, config) -> bool:
    return (isinstance(inputs_sha256, str) and len(inputs_sha256) == 64
            and all(c in "0123456789abcdef" for c in inputs_sha256)
            and isinstance(config, dict) and all(key in config for key in _CONFIG_FIELDS))


def _validate_coordinates(c):
    """Validate the current schema before readers can treat its identity as complete."""
    if type(c.get("identity_complete")) is not bool:
        raise ValueError("coordinate identity_complete must be a boolean")
    complete = c["identity_complete"]
    if complete and not _complete_identity(c.get("inputs_sha256"), c.get("config")):
        raise ValueError("complete coordinates require exact inputs and execution controls")
    for key in ("spec", "methodology", "family", "interface"):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f"coordinate {key} must be a nonempty string")
    if (c.get("repo") is not None and not isinstance(c["repo"], str)
            or complete and (not isinstance(c.get("repo"), str) or not c["repo"].strip())):
        raise ValueError("complete coordinates require a nonempty model repo")
    if not isinstance(c.get("data"), list) or any(not isinstance(x, str) or not x for x in c["data"]):
        raise ValueError("coordinate data must be a list of names")
    for key in ("regimes", "cases"):
        if not isinstance(c.get(key), list) or complete and not c[key]:
            raise ValueError(f"coordinate {key} must be a list, nonempty for complete identity")
    kinds, labels = set(), set()
    for regime in c["regimes"]:
        if not isinstance(regime, dict) or not {"kind", "units", "data", "new_tokens", "aggregate", "data_knobs"} <= regime.keys():
            raise ValueError("coordinate regime is missing required fields")
        if not isinstance(regime["kind"], str) or not regime["kind"]:
            raise ValueError("coordinate regime kind must be a nonempty string")
        if (type(regime["units"]) is not int or regime["units"] < (1 if complete else 0)
                or type(regime["new_tokens"]) is not int or regime["new_tokens"] < 0
                or type(regime["aggregate"]) is not bool or not isinstance(regime["data_knobs"], dict)
                or regime["data"] is not None and not isinstance(regime["data"], str)):
            raise ValueError("invalid coordinate regime settings")
        if complete and (regime["kind"] in kinds or regime["kind"] == "generation" and regime["new_tokens"] == 0):
            raise ValueError("duplicate regime or missing generation length")
        kinds.add(regime["kind"])
    for case in c["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("label"), str) or not case["label"]
                or not isinstance(case.get("params"), dict)):
            raise ValueError("coordinate case requires a label and params dictionary")
        if complete and case["label"] in labels:
            raise ValueError("duplicate coordinate case label")
        labels.add(case["label"])
    if not complete:
        return
    config = c.get("config")
    if (type(config["warmup"]) is not int or config["warmup"] < 0
            or type(config["n_trials"]) is not int or config["n_trials"] < 1
            or not isinstance(config["dtype_control"], str) or not config["dtype_control"]
            or any(not isinstance(config[k], dict) for k in ("hf_kwargs", "vllm_kwargs"))):
        raise ValueError("invalid coordinate timing or model-load controls")
    baseline = config["baseline"]
    if (not isinstance(baseline, dict) or not isinstance(baseline.get("params"), dict)
            or not isinstance(baseline.get("label"), str)):
        raise ValueError("coordinate baseline requires params and label")
    effect = config["effect"]
    if effect is not None and (
            not isinstance(effect, dict)
            or any(not isinstance(effect.get(k), dict) for k in ("baseline_params", "perturbed_params"))
            or any(type(effect.get(k)) not in (int, float) or not math.isfinite(effect[k])
                   for k in ("tv_floor", "top1_ceiling"))):
        raise ValueError("invalid coordinate effect configuration")


def run_coordinates(*, spec: str, methodology: str, family: str, repo: str, interface: str,
                    regimes=(), cases=(), protocol: dict | None = None,
                    protocol_coverage: dict | None = None, inputs_sha256: str | None = None,
                    config: dict | None = None) -> dict:
    """Build a run's `coordinates`: which procedure ran, on what data, under which cell interface.
    Every writer (execute, construct probes, perf sweeps, the manager's job records) builds it
    here, so every reader sees one shape. A `regimes` entry carries kind/units/data plus
    new_tokens/aggregate/data_knobs; a `cases` entry carries its label plus the declared params.
    The params each case actually ran with (cell defaults applied) live in the run's per-cell
    meta, next to its timing."""
    regimes = [{"kind": r["kind"], "units": r["units"], "data": r.get("data"),
                "new_tokens": r.get("new_tokens", 0), "aggregate": r.get("aggregate", True),
                "data_knobs": dict(r.get("data_knobs") or {})} for r in regimes]
    coordinates = deepcopy({
        "schema": COORDINATES_SCHEMA,
        "spec": spec, "methodology": methodology, "family": family, "repo": repo,
        "interface": interface,
        "data": sorted({r["data"] for r in regimes if r["data"]}),
        "regimes": regimes,
        "cases": [{"label": c["label"], "params": dict(c.get("params") or {})} for c in cases],
        "protocol": protocol,
        "protocol_coverage": protocol_coverage,
        "inputs_sha256": inputs_sha256,
        "config": config,
        "identity_complete": _complete_identity(inputs_sha256, config),
    })
    _validate_coordinates(coordinates)
    return coordinates


def spec_coordinates(spec, interface: str, *, inputs_sha256=None, config=None) -> dict:
    """Snapshot an executable spec, using the job contract's exact input-byte encoding."""
    from .jobs.contract import canonical, digest, pack

    if inputs_sha256 is None:
        inputs = b"".join(canonical({"workload": index, "unit": pack(unit)}) + b"\n"
                          for index, regime in enumerate(spec.regimes) for unit in regime.prompts)
        inputs_sha256 = digest(inputs)
    return run_coordinates(
        spec=spec.name, methodology=spec.methodology, family=spec.family, repo=spec.repo,
        interface=interface,
        regimes=[{"kind": r.kind, "units": len(r.prompts), "data": r.data_name,
                  "new_tokens": r.new_tokens, "aggregate": r.aggregate, "data_knobs": r.data_knobs}
                 for r in spec.regimes],
        cases=[{"label": t.label, "params": t.params} for t in spec.tasks],
        protocol=spec.protocol.coordinate() if spec.protocol is not None else None,
        protocol_coverage=spec.protocol_coverage(), inputs_sha256=inputs_sha256,
        config=coordinate_config(spec) if config is None else config)


def procedure(coordinates: dict) -> tuple | None:
    """Correctness comparison identity, or None for incomplete current records.

    Cell interfaces and recorded requirements may differ across intentional backend comparisons.
    Exact inputs, case/regime settings, baseline/effect settings and load options must match.
    Timing controls remain in the artifact and have no role in correctness matching.
    """
    from .jobs.contract import canonical, pack

    if not coordinates:
        return None
    c = read_coordinates(coordinates)
    if not c["identity_complete"]:
        return None
    identity = {"inputs_sha256": c["inputs_sha256"], "data": c["data"],
                "config": {key: c["config"][key] for key in _COMPARISON_CONFIG_FIELDS},
                "regimes": [{k: r[k] for k in ("kind", "units", "data", "new_tokens", "aggregate", "data_knobs")}
                            for r in c["regimes"]],
                "cases": [{"label": case["label"], "params": case["params"]} for case in c["cases"]]}
    return (c["spec"], c["methodology"], c["family"], c["repo"], canonical(pack(identity)))


def read_coordinates(coordinates: dict) -> dict:
    """Validate a current-format coordinate snapshot at an artifact boundary."""
    if not isinstance(coordinates, dict):
        raise ValueError("coordinates must be an object")
    version = coordinates.get("schema")
    if type(version) is not int or version != COORDINATES_SCHEMA:
        raise ValueError(f"unsupported coordinate schema {version!r}; rerun with current code")
    c = deepcopy(coordinates)
    _validate_coordinates(c)
    return c

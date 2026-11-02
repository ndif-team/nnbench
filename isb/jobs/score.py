"""Artifact-only scoring. Reference and comparison axis are explicit, never inferred from names."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .contract import PLAN_VERSION, read_experiment, unpack, validate_result, write_json
from isb.sweep.spec import reference_key
from isb.validation import POLICY_VERSION, performance_eligible


def _load(directory, experiment, backend):
    import torch

    execution = json.loads((directory / "execution.json").read_text())
    if execution["status"] != "completed":
        raise ValueError("job did not complete successfully")
    result = validate_result(directory, experiment, backend)
    artifact = torch.load(directory / "result.pt", map_location="cpu", weights_only=False)
    if artifact["provenance"].get("job", {}).get("experiment_id") != experiment["id"]:
        raise ValueError("tensor artifact has a different experiment identity")
    return result, artifact, execution


def _compatible(candidate, reference, cexec, rexec):
    cmodel = candidate["provenance"].get("model_identity", {})
    rmodel = reference["provenance"].get("model_identity", {})
    for field in ("repo", "revision", "vocab_sha256"):
        if not cmodel.get(field) or cmodel.get(field) != rmodel.get(field):
            raise ValueError(f"unverified or different model/tokenizer {field}")
    if cexec["source"] != rexec["source"]:
        raise ValueError("benchmark source differs between compared jobs")


_ENGINE_START = re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\) with config: (.*)")
_COMPILE = re.compile(r"'mode': <CompilationMode\.(\w+): \d+>")
_GRAPHS = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+)")
_TP = re.compile(r"tensor_parallel_size=(\d+)")


def engine_run(log):
    """What a job's vLLM engine ran, from the engine's own startup line in the job's log
    ("Initializing a V1 LLM engine (v<version>) with config: ..."): {"vllm": version,
    "torch_compile": compilation mode other than CompilationMode.NONE, "cuda_graphs":
    cudagraph_mode other than CUDAGraphMode.NONE, "tensor_parallel_size"}. What ran can differ from
    the declared options: nnsight's taps and interp-engine's static points set
    VLLM_USE_BREAKABLE_CUDAGRAPH, which turns torch.compile off and keeps CUDA graphs on under
    enforce_eager=False. None when the log has no such line (an HF job, or an engine that never
    started); raises when a startup line lacks one of the fields or two lines disagree."""
    log = Path(log)
    if not log.exists():
        return None
    runs = set()
    with log.open(errors="replace") as lines:
        for line in lines:
            start = _ENGINE_START.search(line)
            if start is None:
                continue
            config = start.group(2)
            fields = _COMPILE.search(config), _GRAPHS.search(config), _TP.search(config)
            if not all(fields):
                raise ValueError(f"{log}: engine startup line without compilation mode, "
                                 f"cudagraph_mode or tensor_parallel_size")
            compile_mode, graph_mode, tp = (field.group(1) for field in fields)
            runs.add((start.group(1), compile_mode != "NONE", graph_mode != "NONE", int(tp)))
    if len(runs) > 1:
        raise ValueError(f"{log}: engines started with different configurations: {sorted(runs)}")
    if not runs:
        return None
    version, torch_compile, cuda_graphs, tp = runs.pop()
    return {"vllm": version, "torch_compile": torch_compile, "cuda_graphs": cuda_graphs,
            "tensor_parallel_size": tp}


def _floor_key(run, workload):
    return (run["vllm"], run["torch_compile"], run["cuda_graphs"], run["tensor_parallel_size"],
            workload)


def attach_plain_vllm(rows, results):
    """Whole-system overhead: each timed row on a vLLM engine is divided by the no-intervention
    request of a plain-vLLM job whose engine ran the same vLLM version, torch.compile on or off,
    CUDA graphs on or off and tensor-parallel size, on the same workload. A plain job is one whose
    provenance declares engine mode "plain"; its `__vanilla__` record is the latency. The mode is
    what each engine ran (`engine_run`, from the job's own log), so an eager system's ratio does not
    absorb the CUDA-graph gap, and a system running CUDA graphs without torch.compile, which no
    plain vLLM config runs, gets no `overhead_vs_plain_vllm`. Every row whose mode is not vLLM's
    default (torch.compile and CUDA graphs) also gets `overhead_vs_plain_vllm_default`, against the
    default engine at the same version and tensor-parallel size: the cost a user pays relative to
    plain vLLM as shipped. `results` maps backend -> {"provenance", "auxiliary_calls",
    "engine_run"}; a job without an engine run (an HF job, an older log) matches nothing."""
    plain = {}
    for backend, result in results.items():
        run = result["engine_run"]
        if result["provenance"].get("engine", {}).get("mode") != "plain" or run is None:
            continue
        for call in result.get("auxiliary_calls", []):
            role, workload = call["key"]
            latency = call["record"].get("median_latency_ms")
            if role == "__vanilla__" and latency:
                key = _floor_key(run, workload)
                if key in plain:
                    raise ValueError(f"plain-vLLM jobs {plain[key][0]} and {backend} ran the same "
                                     f"engine mode {key}; keep one floor per mode")
                plain[key] = (backend, latency)
    for row in rows:
        result = results.get(row["backend"])
        if result is None or result["engine_run"] is None or row.get("median_latency_ms") is None:
            continue
        run = result["engine_run"]
        row["engine_run"] = run
        match = plain.get(_floor_key(run, row.get("workload")))
        if match:
            row.update(plain_vllm=match[0], plain_vllm_ms=match[1],
                       overhead_vs_plain_vllm=row["median_latency_ms"] / match[1])
        shipped = {**run, "torch_compile": True, "cuda_graphs": True}
        default = plain.get(_floor_key(shipped, row.get("workload")))
        if run != shipped and default:
            row.update(plain_vllm_default=default[0], plain_vllm_default_ms=default[1],
                       overhead_vs_plain_vllm_default=row["median_latency_ms"] / default[1])


def attach_worker_memory(rows, results):
    """Each row's engine-worker peak against its own system's no-intervention request in the same
    regime (`__vanilla__`): `worker_peak_mem_mb_vanilla` and `worker_peak_mem_delta_mb`, both the
    max over ranks. `results` maps backend -> {"auxiliary_calls"}."""
    vanilla = {}
    for backend, result in results.items():
        for call in result.get("auxiliary_calls", []):
            role, regime = call["key"]
            peak = call["record"].get("worker_peak_mem_mb_max")
            if role == "__vanilla__" and peak is not None:
                vanilla[(backend, regime)] = peak
    for row in rows:
        base = vanilla.get((row["backend"], row.get("workload")))
        if base is not None and row.get("worker_peak_mem_mb_max") is not None:
            row.update(worker_peak_mem_mb_vanilla=base,
                       worker_peak_mem_delta_mb=round(row["worker_peak_mem_mb_max"] - base, 1))


def score_experiment(directory, backends, reference=None, comparison="correctness"):
    import torch
    from isb.oracle.equivalence import compare, is_equivalent

    if comparison not in {"correctness", "equivalence"}:
        raise ValueError("unknown comparison policy")
    directory = Path(directory)
    experiment = read_experiment(directory)
    loaded, failures = {}, {}
    for backend in backends:
        try:
            loaded[backend] = _load(directory / backend, experiment, backend)
        except Exception as error:
            failures[backend] = str(error)
    rows = []
    for backend in backends:
        if backend not in loaded:
            rows.append({"backend": backend, "state": "JOB_FAILED", "error": failures[backend]})
            continue
        result, artifact, execution = loaded[backend]
        ref = loaded.get(reference) if backend != reference else None
        incompatible = None
        if ref:
            try:
                _compatible(artifact, ref[1], execution, ref[2])
            except ValueError as error:
                incompatible = str(error)
        spec = unpack(experiment["spec"])
        for cell in result["cells"]:
            row = {"backend": backend, **cell}
            key = (cell["workload"], cell["label"])
            if cell["state"] != "RAN":
                rows.append(row)
                continue
            actual = artifact["outputs"].get(key)
            if (not isinstance(actual, torch.Tensor) or not actual.numel()
                    or not torch.isfinite(actual).all()):
                row.update(state="INVALID_OUTPUT", error="missing, empty or nonfinite tensor output")
                rows.append(row)
                continue
            if reference is None or backend == reference:
                rows.append(row)
                continue
            if incompatible:
                row.update(state="INCOMPATIBLE", error=incompatible)
            elif not ref:
                row.update(state="NO_REFERENCE", error=failures.get(reference))
            else:
                # An alternative realization is scored against the reference's default for its row.
                row_label = cell.get("row", cell["label"])
                refkey = reference_key(key[0], row_label, comparison)
                expected = ref[1]["outputs"].get(refkey)
                actual = artifact["outputs"].get(key)
                effect = ref[1]["outputs"].get(("__meta__",), {}).get(("__effect__", key[0]))
                if expected is None:
                    row["state"] = "NO_REFERENCE"
                elif (spec["effect"] and key[0] in {"interactive", "generation"}
                      and (not effect or not effect.get("strong"))):
                    row.update(state="INVALID_REFERENCE", error="missing or weak intervention effect")
                elif not isinstance(expected, torch.Tensor):
                    row.update(state="INVALID_REFERENCE", error="expected tensor reference")
                else:
                    # Only align documented vocabulary padding; never truncate arbitrary axes.
                    size = artifact["provenance"]["model_identity"]["vocab_size"]
                    padded_sizes = {math.ceil(size / block) * block for block in (64, 128, 256)}
                    if (actual.ndim and expected.ndim and expected.shape[-1] == size
                            and actual.shape[-1] in padded_sizes):
                        actual = actual[..., :size]
                    if not expected.numel() or not torch.isfinite(expected).all():
                        row.update(state="INVALID_REFERENCE", error="empty/nonfinite reference output")
                    elif (actual.shape != expected.shape or not actual.numel()
                          or not torch.isfinite(actual).all()):
                        row.update(state="INVALID_OUTPUT",
                                   error="shape mismatch or empty/nonfinite output")
                    else:
                        metrics = compare(expected, actual)
                        good, bad = (("SUPPORTED", "NUMERICAL_MISMATCH") if comparison == "correctness"
                                     else ("EQUIVALENT", "DIVERGENT"))
                        row.update(state=good if is_equivalent(metrics) else bad, metrics=metrics)
            rows.append(row)
    calls = {backend: {"provenance": artifact["provenance"],
                       "auxiliary_calls": result.get("auxiliary_calls", []),
                       "engine_run": engine_run(directory / backend / "execution.log")}
             for backend, (result, artifact, _) in loaded.items()}
    attach_plain_vllm(rows, calls)
    attach_worker_memory(rows, calls)
    for row in rows:
        row["performance_eligible"] = performance_eligible(row["state"])
        row["validation_state"] = (
            "OUTPUT_CONTRACT_FAILED" if row["state"] == "INVALID_OUTPUT" else
            "NUMERICAL_CHECK_PASSED" if row["state"] in {"SUPPORTED", "EQUIVALENT"} else
            "UNRESOLVED" if row["performance_eligible"] else "EXECUTION_FAILED"
        )
    return {"validation_policy_version": POLICY_VERSION,
            "experiment_id": experiment["id"], "spec": experiment["spec"]["name"],
            "reference": reference, "comparison": comparison, "cells": rows}


def score_run(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    reports = [score_experiment(directory / "experiments" / name, plan["backends"],
                                plan["reference"], plan["comparison"])
               for name in plan["experiments"]]
    report = {"version": PLAN_VERSION, "run": directory.name, "experiments": reports,
              "not_applicable": plan.get("not_applicable", [])}
    write_json(directory / "report.json", report)
    for experiment in reports:
        for row in experiment["cells"]:
            print(f"{experiment['spec']} | {row['backend']} | {row.get('workload', '-')} | "
                  f"{row.get('label', '-')} | {row['state']}")
    for row in report["not_applicable"]:
        print(f"{row['case']} | - | - | {row['label']} | NOT_APPLICABLE ({row['reason']})")
    return report

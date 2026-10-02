"""Artifact-only scoring. Reference and comparison axis are explicit, never inferred from names."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contract import PLAN_VERSION, read_experiment, unpack, validate_result, write_json
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


def _eager(provenance):
    """Whether the job's engine ran eager (no torch.compile / CUDA graphs), as its backend recorded."""
    return bool(provenance.get("engine", {}).get("params", {}).get("enforce_eager", False))


def attach_plain_vllm(rows, results):
    """Whole-system overhead (design.md §12.14): each timed row on a vLLM engine is divided by the
    no-intervention request of a plain-vLLM job at the same vLLM version, execution mode (eager or
    compiled) and workload. A plain job is one whose provenance declares engine mode "plain"; its
    `__vanilla__` record is the latency. Matching the mode keeps an eager-only system's ratio from
    absorbing the eager-vs-CUDA-graph gap; `overhead_vs_plain_vllm_default` additionally divides
    an eager row by vLLM's default compiled engine when that job is present, the cost a user pays
    relative to plain vLLM as shipped. `results` maps backend -> {"provenance", "auxiliary_calls"};
    a provenance without an engine record (older artifacts) matches nothing."""
    plain = {}
    for backend, result in results.items():
        provenance = result["provenance"]
        if provenance.get("engine", {}).get("mode") != "plain":
            continue
        for call in result.get("auxiliary_calls", []):
            role, workload = call["key"]
            latency = call["record"].get("median_latency_ms")
            if role == "__vanilla__" and latency:
                key = (provenance["client"].get("vllm"), _eager(provenance), workload)
                plain[key] = (backend, latency)
    for row in rows:
        result = results.get(row["backend"])
        if result is None or row.get("median_latency_ms") is None:
            continue
        provenance = result["provenance"]
        if provenance.get("engine", {}).get("kind") != "vllm":
            continue
        version, eager = provenance["client"].get("vllm"), _eager(provenance)
        match = plain.get((version, eager, row.get("workload")))
        if match:
            row.update(plain_vllm=match[0], plain_vllm_ms=match[1],
                       overhead_vs_plain_vllm=row["median_latency_ms"] / match[1])
        default = plain.get((version, False, row.get("workload")))
        if eager and default:
            row.update(plain_vllm_default=default[0], plain_vllm_default_ms=default[1],
                       overhead_vs_plain_vllm_default=row["median_latency_ms"] / default[1])


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
                refkey = (("batched_perprompt", key[1]) if key[0] == "batched"
                          and comparison == "correctness" else key)
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
    attach_plain_vllm(rows, {backend: {"provenance": artifact["provenance"],
                                       "auxiliary_calls": result.get("auxiliary_calls", [])}
                             for backend, (result, artifact, _) in loaded.items()})
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
    report = {"version": PLAN_VERSION, "run": directory.name, "experiments": reports}
    write_json(directory / "report.json", report)
    for experiment in reports:
        for row in experiment["cells"]:
            print(f"{experiment['spec']} | {row['backend']} | {row.get('workload', '-')} | "
                  f"{row.get('label', '-')} | {row['state']}")
    return report

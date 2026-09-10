"""Artifact-only scoring. Reference and comparison axis are explicit, never inferred from names."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contract import PLAN_VERSION, read_experiment, unpack, validate_result, write_json


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
                elif not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
                    row.update(state="ERROR", error="expected tensor outputs")
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
                        row.update(state="SILENTLY_WRONG" if comparison == "correctness" else "DIVERGENT",
                                   error="shape mismatch or empty/nonfinite output")
                    else:
                        metrics = compare(expected, actual)
                        good, bad = (("SUPPORTED", "SILENTLY_WRONG") if comparison == "correctness"
                                     else ("EQUIVALENT", "DIVERGENT"))
                        row.update(state=good if is_equivalent(metrics) else bad, metrics=metrics)
            rows.append(row)
    return {"experiment_id": experiment["id"], "spec": experiment["spec"]["name"],
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

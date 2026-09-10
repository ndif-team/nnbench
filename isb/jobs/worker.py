"""Shared nnsight experiment worker; backend directories supply their own constructors."""
from __future__ import annotations

import json
import os
import random
import tempfile
import traceback
from pathlib import Path

from .contract import VERSION, check_experiment, expected_cells, file_digest, restore_spec, write_json


def merge_requirements(required, configured):
    for key in required.keys() & configured.keys():
        if required[key] != configured[key]:
            raise ValueError(f"backend option {key!r} conflicts with model requirement")
    return {**required, **configured}


def main(create_backend, job=Path("/job"), output=Path("/output")):
    """create_backend(spec) returns (backend instance, descriptive RunConfig)."""
    job, output = Path(job), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"version": VERSION}
    if os.environ.get("ISB_BACKEND"):
        identity["backend"] = os.environ["ISB_BACKEND"]
    published = False
    try:
        # Establish only checksum-verified identity before restoration. Malformed jobs, input
        # checksum failures, dependency imports, and spec restoration share the failure path.
        experiment = json.loads((job / "experiment.json").read_text())
        check_experiment(experiment)
        identity["experiment_id"] = experiment["id"]
        input_hash = experiment.get("inputs_sha256")
        if (isinstance(input_hash, str) and len(input_hash) == 64
                and all(c in "0123456789abcdef" for c in input_hash)):
            identity["inputs_sha256"] = input_hash
        if "backend" not in identity:
            raise ValueError("ISB_BACKEND must identify the worker")
        spec, experiment = restore_spec(job)

        import torch
        import isb.methodologies  # noqa: F401  # register explicit intervention functions
        from isb.runs import resolve_provenance
        from isb.sweep.execute import execute_run

        random.seed(experiment["seed"])
        torch.manual_seed(experiment["seed"])
        backend, run = create_backend(spec)
        provenance = {**resolve_provenance(run), "job": identity, "experiment": experiment}
        with tempfile.TemporaryDirectory(prefix=".partial-", dir=output) as staging:
            partial = Path(staging)
            execute_run(spec, run, str(partial), "result", backend=backend,
                        interface=backend.name, provenance=provenance)
            artifact = torch.load(partial / "result.pt", map_location="cpu", weights_only=False)
            outputs, provenance = artifact["outputs"], artifact["provenance"]
            metadata = outputs.get(("__meta__",), {})
            cells = []
            requested = expected_cells(experiment)
            for key in sorted(requested):
                info = dict(metadata.get(key, {}))
                error = info.get("error")
                if not error and outputs.get(key) is None:
                    error = "cell returned no output"
                info["error"] = error
                metadata[key] = info
                cells.append({"workload": key[0], "label": key[1],
                              "state": "ERROR" if error else "RAN", **info})
            auxiliary_calls = [{"key": list(key), "record": info}
                               for key, info in metadata.items() if key not in requested]
            outputs[("__meta__",)] = metadata
            torch.save(artifact, partial / "result.pt")
            os.replace(partial / "result.pt", output / "result.pt")
            published = True
        write_json(output / "result.json", {
            **identity, "status": "completed", "cells": cells, "provenance": provenance,
            "auxiliary_calls": auxiliary_calls,
            "outputs_sha256": file_digest(output / "result.pt"),
        })
    except Exception as error:
        traceback.print_exc()
        if published:
            (output / "result.pt").unlink(missing_ok=True)
        write_json(output / "result.json", {**identity, "status": "failed", "error": repr(error)})
        raise


def options():
    value = json.loads(os.environ.get("ISB_ENGINE_OPTIONS", "{}"))
    if not isinstance(value, dict):
        raise ValueError("ISB_ENGINE_OPTIONS must be a JSON object")
    return value

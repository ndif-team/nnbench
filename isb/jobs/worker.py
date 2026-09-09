"""Shared nnsight experiment worker; backend directories supply their own constructors."""
from __future__ import annotations

import json
import os
import random
import traceback
from pathlib import Path

from .contract import VERSION, expected_cells, file_digest, restore_spec, write_json


def merge_requirements(required, configured):
    for key in required.keys() & configured.keys():
        if required[key] != configured[key]:
            raise ValueError(f"backend option {key!r} conflicts with model requirement")
    return {**required, **configured}


def main(create_backend, job=Path("/job"), output=Path("/output")):
    """create_backend(spec) returns (backend instance, descriptive RunConfig)."""
    import torch
    import isb.methodologies  # noqa: F401 — register explicit intervention functions
    from isb.runs import resolve_provenance
    from isb.sweep.execute import execute_run

    output.mkdir(parents=True, exist_ok=True)
    spec, experiment = restore_spec(job)
    backend_name = os.environ["ISB_BACKEND"]
    identity = {"version": VERSION, "experiment_id": experiment["id"],
                "inputs_sha256": experiment["inputs_sha256"], "backend": backend_name}
    try:
        random.seed(experiment["seed"])
        torch.manual_seed(experiment["seed"])
        backend, run = create_backend(spec)
        provenance = {**resolve_provenance(run), "job": identity, "experiment": experiment}
        partial = output / ".partial"
        partial.mkdir()
        execute_run(spec, run, str(partial), "result", backend=backend,
                    interface=backend.name, provenance=provenance)
        artifact = torch.load(partial / "result.pt", map_location="cpu", weights_only=False)
        outputs, provenance = artifact["outputs"], artifact["provenance"]
        metadata = outputs.get(("__meta__",), {})
        cells = []
        for key in sorted(expected_cells(experiment)):
            info = dict(metadata.get(key, {}))
            error = info.get("error")
            if not error and outputs.get(key) is None:
                error = "cell returned no output"
            info["error"] = error
            metadata[key] = info
            cells.append({"workload": key[0], "label": key[1],
                          "state": "ERROR" if error else "RAN", **info})
        outputs[("__meta__",)] = metadata
        torch.save(artifact, partial / "result.pt")
        os.replace(partial / "result.pt", output / "result.pt")
        partial.rmdir()
        write_json(output / "result.json", {
            **identity, "status": "completed", "cells": cells, "provenance": provenance,
            "outputs_sha256": file_digest(output / "result.pt"),
        })
    except Exception as error:
        traceback.print_exc()
        write_json(output / "result.json", {**identity, "status": "failed", "error": repr(error)})
        raise


def options():
    value = json.loads(os.environ.get("ISB_ENGINE_OPTIONS", "{}"))
    if not isinstance(value, dict):
        raise ValueError("ISB_ENGINE_OPTIONS must be a JSON object")
    return value

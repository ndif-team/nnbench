"""CPU-only executable implementing the minimal backend artifact contract for Docker tests."""
import os
import sys
import time
from pathlib import Path

from isb.jobs.contract import expected_cells, file_digest, read_experiment, write_json

if sys.argv[1] == "crash":
    raise SystemExit(7)
if sys.argv[1] == "timeout":
    time.sleep(120)

import torch  # noqa: E402

experiment = read_experiment(Path("/job"))
backend = os.environ["ISB_BACKEND"]
identity = {"version": 1, "experiment_id": experiment["id"], "backend": backend,
            "inputs_sha256": experiment["inputs_sha256"]}
outputs = {key: torch.ones(1, 8) for key in expected_cells(experiment)}
torch.save({"outputs": outputs, "provenance": {"job": identity}}, "/output/result.pt")
write_json("/output/result.json", {**identity, "status": "completed",
    "cells": [{"workload": k[0], "label": k[1], "state": "RAN"} for k in outputs],
    "outputs_sha256": file_digest("/output/result.pt")})

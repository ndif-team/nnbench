"""Independent CPU worker: stdlib input reader, PyTorch artifact writer, no nnbench imports."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            return tuple(unpack(v) for v in value["$tuple"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value

if sys.argv[1] == "crash":
    raise SystemExit(7)
if sys.argv[1] == "timeout":
    time.sleep(120)

job = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/job")
output = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/output")
experiment = json.loads((job / "experiment.json").read_text())
supported = (1,) if sys.argv[1] == "v1-only" else (2,)
if type(experiment.get("version")) is not int or experiment["version"] not in supported:
    raise SystemExit(f"unsupported experiment version {experiment.get('version')}; supported: {supported}")
assert digest(canonical({k: v for k, v in experiment.items() if k != "id"})) == experiment["id"]
assert digest((job / "inputs.jsonl").read_bytes()) == experiment["inputs_sha256"]
spec = unpack(experiment["spec"])
expected = {(r["kind"], t["label"]) for r in spec["regimes"] for t in spec["tasks"]}

import torch  # noqa: E402

backend = os.environ["ISB_BACKEND"]
identity = {"version": experiment["version"], "experiment_id": experiment["id"], "backend": backend,
            "inputs_sha256": experiment["inputs_sha256"]}
outputs = {key: torch.ones(1, 8) for key in sorted(expected)}
output.mkdir(parents=True, exist_ok=True)
torch.save({"outputs": outputs, "provenance": {"job": identity}}, output / "result.pt")
result = {**identity, "status": "completed",
    "cells": [{"workload": k[0], "label": k[1], "state": "RAN"} for k in outputs],
    "outputs_sha256": digest((output / "result.pt").read_bytes())}
(output / "result.json.tmp").write_bytes(canonical(result) + b"\n")
os.replace(output / "result.json.tmp", output / "result.json")

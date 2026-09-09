"""The on-disk experiment/result contract shared by launcher and backend workers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

VERSION = 1


def pack(value):
    # Preserve paired inputs and tuple-valued task parameters through JSON.
    if isinstance(value, tuple):
        return {"$tuple": [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            return tuple(unpack(v) for v in value["$tuple"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def prepare(spec, directory, *, seed=0):
    """Materialize the experiment once; containers never look up a spec or dataset."""
    directory = Path(directory)
    coverage = spec.protocol_coverage()
    record = asdict(spec)
    # Keep the version-1 execution wire format understood by independent backend workers.
    # The richer Python types travel as optional, identity-covered description metadata.
    description = {"protocol_template": record.pop("protocol"), "tasks": record.pop("tasks"),
                   "protocol_coverage": coverage}
    record.pop("protocol_absence_reason")
    record["workloads"] = record.pop("regimes")
    record["tasks"] = [(task.params, task.label) for task in spec.tasks]
    if not spec.regimes:
        raise ValueError("experiment must contain a workload")
    records = []
    kinds = set()
    labels = [task.label for task in spec.tasks]
    if not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task labels must be nonempty and unique")
    for index, workload in enumerate(record["workloads"]):
        if workload["kind"] in kinds:
            raise ValueError("duplicate workload kinds would collide in output cell keys")
        kinds.add(workload["kind"])
        units = workload.pop("prompts")
        if not units:
            raise ValueError("empty workload")
        workload["units"] = len(units)
        records.extend({"workload": index, "unit": pack(unit)} for unit in units)
    if spec.n_trials < 1 or spec.warmup < 0:
        raise ValueError("n_trials must be positive and warmup nonnegative")
    inputs = b"".join(canonical(row) + b"\n" for row in records)
    experiment = {"version": VERSION, "spec": pack(record), "seed": seed,
                  "inputs_sha256": digest(inputs), "description": pack(description)}
    experiment["id"] = digest(canonical(experiment))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs.jsonl").write_bytes(inputs)
    write_json(directory / "experiment.json", experiment)
    return experiment


def read_experiment(directory):
    directory = Path(directory)
    experiment = json.loads((directory / "experiment.json").read_text())
    body = {k: v for k, v in experiment.items() if k != "id"}
    if experiment.get("version") != VERSION or digest(canonical(body)) != experiment.get("id"):
        raise ValueError("invalid experiment version or checksum")
    if file_digest(directory / "inputs.jsonl") != experiment["inputs_sha256"]:
        raise ValueError("input checksum mismatch")
    return experiment


def restore_spec(directory):
    from isb.protocol import InterventionSpec, protocol_for
    from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime, TaskSpec

    experiment = read_experiment(directory)
    record = unpack(experiment["spec"])
    workloads = record.pop("workloads")
    units = [[] for _ in workloads]
    for line in (Path(directory) / "inputs.jsonl").read_text().splitlines():
        row = json.loads(line)
        units[row["workload"]].append(unpack(row["unit"]))
    for index, workload in enumerate(workloads):
        if workload.pop("units") != len(units[index]):
            raise ValueError("workload input count mismatch")
    record["regimes"] = [ExecutionRegime(prompts=u, **w) for w, u in zip(workloads, units)]
    coverage = None
    if "description" in experiment:
        description = unpack(experiment["description"])
        template = description["protocol_template"]
        record["protocol"] = InterventionSpec(**template) if template is not None else None
        tasks = [TaskSpec(**task) for task in description["tasks"]]
        if [(task.params, task.label) for task in tasks] != record["tasks"]:
            raise ValueError("task description does not match executable parameters")
        record["tasks"] = tasks
        coverage = description.get("protocol_coverage")
        if "protocol_coverage" in description:
            if not isinstance(coverage, dict):
                raise ValueError("invalid protocol coverage")
            if coverage.get("status") == "undescribed":
                record["protocol_absence_reason"] = coverage.get("reason")
            elif coverage != {"status": "described"} or template is None:
                raise ValueError("invalid protocol coverage")
    if coverage is None and record.get("protocol") is None and protocol_for(record["methodology"]) is None:
        record["protocol_absence_reason"] = "Legacy experiment predates explicit protocol coverage"
    record["baseline"] = BaselineSpec(**record["baseline"])
    if record["effect"] is not None:
        record["effect"] = EffectSpec(**record["effect"])
    spec = CellConfig(**record)
    if coverage is not None and spec.protocol_coverage() != coverage:
        raise ValueError("protocol coverage does not match descriptor")
    return spec, experiment


def expected_cells(experiment):
    spec = unpack(experiment["spec"])
    return {(w["kind"], label) for w in spec["workloads"] for _, label in spec["tasks"]}


def validate_result(directory, experiment, backend):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if (result.get("version") != VERSION or result.get("status") != "completed"
            or result.get("experiment_id") != experiment["id"]
            or result.get("backend") != backend
            or result.get("inputs_sha256") != experiment["inputs_sha256"]):
        raise ValueError("result identity/status does not match requested job")
    cells = result.get("cells", [])
    keys = [(row["workload"], row["label"]) for row in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected_cells(experiment):
        raise ValueError("result does not account for every requested cell exactly once")
    if any(row.get("state") not in {"RAN", "ERROR", "UNSUPPORTED"} for row in cells):
        raise ValueError("invalid execution cell status")
    if file_digest(directory / "result.pt") != result.get("outputs_sha256"):
        raise ValueError("missing or corrupted tensor artifact")
    return result

"""The on-disk experiment/result contract shared by launcher and backend workers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

VERSION = 2            # experiment/result files this code writes
VERSIONS = (1, 2)      # experiment/result files this code reads (1: `workloads` + task tuples)
PLAN_VERSION = 1       # the launcher's plan.json / report.json
LEGACY_PROTOCOL_REASON = "Legacy experiment predates explicit protocol coverage"


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
    """Materialize the experiment once; containers never look up a spec or dataset. The spec
    record is the dataclass as written: regimes, tasks as {label, params}, the protocol template
    and, for an undescribed methodology, its absence reason. All of it is covered by the id."""
    directory = Path(directory)
    spec.protocol_coverage()                       # re-validate: specs are editable after construction
    record = asdict(spec)
    # Authored/restored validation policy is in-memory. The legacy marker preserves genuinely
    # missing historical metadata when an old recipe is submitted as a new v2 job.
    if record.get("protocol_source") != "legacy":
        record.pop("protocol_source", None)
    if not spec.regimes:
        raise ValueError("experiment must contain a workload")
    records = []
    kinds = set()
    labels = [task["label"] for task in record["tasks"]]
    if not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task labels must be nonempty and unique")
    for index, regime in enumerate(record["regimes"]):
        if regime["kind"] in kinds:
            raise ValueError("duplicate workload kinds would collide in output cell keys")
        kinds.add(regime["kind"])
        units = regime.pop("prompts")
        if not units:
            raise ValueError("empty workload")
        regime["units"] = len(units)
        records.extend({"workload": index, "unit": pack(unit)} for unit in units)
    if spec.n_trials < 1 or spec.warmup < 0:
        raise ValueError("n_trials must be positive and warmup nonnegative")
    inputs = b"".join(canonical(row) + b"\n" for row in records)
    experiment = {"version": VERSION, "spec": pack(record), "seed": seed,
                  "inputs_sha256": digest(inputs)}
    experiment["id"] = digest(canonical(experiment))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs.jsonl").write_bytes(inputs)
    write_json(directory / "experiment.json", experiment)
    return experiment


def check_experiment(experiment):
    """Identity check shared by every reader of an experiment.json."""
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a JSON object")
    body = {k: v for k, v in experiment.items() if k != "id"}
    if (type(experiment.get("version")) is not int or experiment["version"] not in VERSIONS
            or digest(canonical(body)) != experiment.get("id")):
        raise ValueError("invalid experiment version or checksum")


def read_experiment(directory):
    directory = Path(directory)
    experiment = json.loads((directory / "experiment.json").read_text())
    check_experiment(experiment)
    if file_digest(directory / "inputs.jsonl") != experiment["inputs_sha256"]:
        raise ValueError("input checksum mismatch")
    return experiment


def wire_spec(experiment):
    """Return the frozen spec in the current shape without changing experiment identity.

    Version one has two historical forms: executable tuples alone, and those same tuples plus
    identity-covered protocol/task descriptions. The saved description owns the latter's
    metadata. A description-free record keeps its missing metadata visible, including when
    today's protocol table happens to contain its methodology.
    """
    if type(experiment.get("version")) is not int or experiment["version"] not in VERSIONS:
        raise ValueError("unsupported experiment version")
    record = unpack(experiment["spec"])
    if not isinstance(record, dict):
        raise ValueError("experiment spec must be an object")
    source = record.get("protocol_source", "restored")
    if source not in {"authored", "restored", "legacy"}:
        raise ValueError("invalid saved protocol source")
    record["protocol_source"] = "legacy" if source == "legacy" else "restored"
    if experiment["version"] == 1:
        record["regimes"] = record.pop("workloads")
        record["tasks"] = [{"label": label, "params": params} for params, label in record["tasks"]]
        if "description" in experiment:
            description = unpack(experiment["description"])
            if not isinstance(description, dict):
                raise ValueError("invalid experiment description")
            template = description["protocol_template"]
            described_tasks = []
            for task in description["tasks"]:
                semantics, realization = task.get("semantics", {}), task.get("realization", {})
                if not isinstance(semantics, dict) or not isinstance(realization, dict):
                    raise ValueError("invalid task description parameters")
                if semantics.keys() & realization.keys():
                    raise ValueError("task description parameters overlap")
                described_tasks.append({"label": task["label"], "params": {**semantics, **realization}})
            if described_tasks != record["tasks"]:
                raise ValueError("task description does not match executable parameters")
            record["protocol"] = template
            coverage = description.get("protocol_coverage")
            if "protocol_coverage" not in description:
                # The earliest descriptions predate explicit coverage. A saved template still
                # describes the method; absent templates carry no known opt-out policy.
                record["protocol_absence_reason"] = None if template is not None else LEGACY_PROTOCOL_REASON
                if template is None:
                    record["protocol_source"] = "legacy"
            elif coverage == {"status": "described"} and template is not None:
                record["protocol_absence_reason"] = None
            elif (isinstance(coverage, dict) and coverage.get("status") == "undescribed"
                  and set(coverage) == {"status", "reason"} and template is None
                  and isinstance(coverage["reason"], str) and coverage["reason"].strip()):
                record["protocol_absence_reason"] = coverage["reason"]
            else:
                raise ValueError("protocol coverage does not match descriptor")
        else:
            record["protocol"] = None
            record["protocol_absence_reason"] = LEGACY_PROTOCOL_REASON
            record["protocol_source"] = "legacy"
    if not isinstance(record.get("tasks"), list) or not isinstance(record.get("regimes"), list):
        raise ValueError("experiment requires task and regime lists")
    labels = [task["label"] for task in record["tasks"]]
    if not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task labels must be nonempty and unique")
    if any(not isinstance(task.get("params"), dict) for task in record["tasks"]):
        raise ValueError("task parameters must be objects")
    kinds = [regime["kind"] for regime in record["regimes"]]
    if not kinds or len(kinds) != len(set(kinds)):
        raise ValueError("experiment requires unique nonempty regimes")
    return record


def restore_spec(directory):
    from isb.protocol import InterventionSpec
    from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime, TaskSpec

    experiment = read_experiment(directory)
    record = wire_spec(experiment)
    regimes = record.pop("regimes")
    units = [[] for _ in regimes]
    for line in (Path(directory) / "inputs.jsonl").read_text().splitlines():
        row = json.loads(line)
        index = row["workload"]
        if type(index) is not int or not 0 <= index < len(regimes):
            raise ValueError("input workload index is out of range")
        units[index].append(unpack(row["unit"]))
    for index, regime in enumerate(regimes):
        if regime.pop("units") != len(units[index]):
            raise ValueError("workload input count mismatch")
    record["regimes"] = [ExecutionRegime(prompts=u, **w) for w, u in zip(regimes, units)]
    record["tasks"] = [TaskSpec(**task) for task in record["tasks"]]
    if record["protocol"] is not None:
        record["protocol"] = InterventionSpec(**record["protocol"])
    record["baseline"] = BaselineSpec(**record["baseline"])
    if record["effect"] is not None:
        record["effect"] = EffectSpec(**record["effect"])
    return CellConfig(**record), experiment


def expected_cells(experiment):
    spec = wire_spec(experiment)
    return {(w["kind"], task["label"]) for w in spec["regimes"] for task in spec["tasks"]}


def validate_result(directory, experiment, backend):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if (result.get("version") not in VERSIONS or result.get("status") != "completed"
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

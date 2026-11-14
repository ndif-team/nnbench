"""The on-disk experiment/result contract shared by launcher and backend workers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from isb.sweep.spec import ALL_LOAD_FIELDS, regime_kind, regime_name

from .schema import (WIRE_READ, WIRE_VERSION, drop_saturated_floors, require, upgrade_calls,
                     upgrade_rows)


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


def encode_inputs(regime_units):
    """inputs.jsonl: one line per unit, {"regime": its regime's index, "unit": the unit}, regimes in
    order; the bytes an experiment's `inputs_sha256` covers."""
    return b"".join(canonical({"regime": index, "unit": pack(unit)}) + b"\n"
                    for index, units in enumerate(regime_units) for unit in units)


def prepare(spec, directory, *, seed=0):
    """Materialize the experiment once; containers never look up a spec or dataset. The spec
    record is the dataclass as written: regimes, tasks as {label, params}, the protocol template
    and, for an undescribed methodology, its absence reason. All of it is covered by the id."""
    directory = Path(directory)
    spec.protocol_coverage()                       # re-validate: specs are editable after construction
    record = asdict(spec)
    # Authoring/restoration validation policy belongs to the in-memory spec.
    record.pop("protocol_source", None)
    if not spec.regimes:
        raise ValueError("experiment must contain a regime")
    regime_units = []
    kinds = set()
    labels = [task["label"] for task in record["tasks"]]
    if not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task labels must be nonempty and unique")
    for index, regime in enumerate(record["regimes"]):
        name = regime_name(regime["kind"], regime)
        if name in kinds:
            raise ValueError("duplicate regimes would collide in output cell keys")
        kinds.add(name)
        for key in set(ALL_LOAD_FIELDS) - set(regime_kind(regime["kind"]).load):
            regime.pop(key, None)     # each regime records only its own pattern's settings
        units = regime.pop("units")
        if not units:
            raise ValueError(f"regime {name} has no units")
        regime["units"] = len(units)
        regime_units.append(units)
    if spec.n_trials < 0 or spec.warmup < 0:     # n_trials=0: a check run, nothing timed
        raise ValueError("n_trials and warmup must be nonnegative")
    inputs = encode_inputs(regime_units)
    experiment = {"version": WIRE_VERSION, "spec": pack(record), "seed": seed,
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
    require(experiment.get("version"), WIRE_READ, "experiment")
    if digest(canonical(body)) != experiment.get("id"):
        raise ValueError("invalid experiment checksum")


def read_experiment(directory):
    directory = Path(directory)
    experiment = json.loads((directory / "experiment.json").read_text())
    check_experiment(experiment)
    if file_digest(directory / "inputs.jsonl") != experiment["inputs_sha256"]:
        raise ValueError("input checksum mismatch")
    return experiment


def wire_spec(experiment):
    """Validate the frozen spec and retain its saved descriptor."""
    require(experiment.get("version"), WIRE_READ, "experiment")
    record = unpack(experiment["spec"])
    if not isinstance(record, dict):
        raise ValueError("experiment spec must be an object")
    record["protocol_source"] = "restored"
    if not isinstance(record.get("tasks"), list) or not isinstance(record.get("regimes"), list):
        raise ValueError("experiment requires task and regime lists")
    labels = [task["label"] for task in record["tasks"]]
    if not labels or not all(isinstance(label, str) and label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task labels must be nonempty and unique")
    if any(not isinstance(task.get("params"), dict) for task in record["tasks"]):
        raise ValueError("task parameters must be objects")
    kinds = [regime_name(regime["kind"], regime)
             for regime in record["regimes"]]
    if not kinds or len(kinds) != len(set(kinds)):
        raise ValueError("experiment requires unique nonempty regimes")
    return record


def restore_spec(directory):
    from isb.protocol import InterventionSpec
    from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime, TaskSpec

    experiment = read_experiment(directory)
    require(experiment["version"], (WIRE_VERSION,), "experiment")     # a worker runs current ones
    record = wire_spec(experiment)
    regimes = record.pop("regimes")
    units = [[] for _ in regimes]
    for line in (Path(directory) / "inputs.jsonl").read_text().splitlines():
        row = json.loads(line)
        index = row["regime"]
        if type(index) is not int or not 0 <= index < len(regimes):
            raise ValueError("input regime index is out of range")
        units[index].append(unpack(row["unit"]))
    for index, regime in enumerate(regimes):
        if regime.pop("units") != len(units[index]):
            raise ValueError("regime input count mismatch")
    record["regimes"] = [ExecutionRegime(units=u, **w) for w, u in zip(regimes, units)]
    record["tasks"] = [TaskSpec(**task) for task in record["tasks"]]
    if record["protocol"] is not None:
        record["protocol"] = InterventionSpec(**record["protocol"])
    record["baseline"] = BaselineSpec(**record["baseline"])
    if record["effect"] is not None:
        record["effect"] = EffectSpec(**record["effect"])
    return CellConfig(**record), experiment


def expected_cells(experiment):
    spec = wire_spec(experiment)
    return {(regime_name(w["kind"], w), task["label"])
            for w in spec["regimes"] for task in spec["tasks"]}


def validate_result(directory, experiment, backend):
    """The job's result.json, checked against the experiment, its cells and auxiliary calls in the
    current fields. A cell carrying `row` and `alternative` is an alternative realization of that
    row; a result without such cells holds the default realizations only."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    require(result.get("version"), WIRE_READ, "result")
    if (result.get("status") != "completed"
            or result.get("experiment_id") != experiment["id"]
            or result.get("backend") != backend
            or result.get("inputs_sha256") != experiment["inputs_sha256"]):
        raise ValueError("result identity/status does not match requested job")
    calls = upgrade_calls(result.get("auxiliary_calls", []), result["version"])
    cells = upgrade_rows(result.get("cells", []), result["version"], WIRE_VERSION)
    result["cells"], result["auxiliary_calls"] = drop_saturated_floors(cells, calls), calls
    keys = [(row["regime"], row["label"]) for row in cells]
    expected = expected_cells(experiment)
    defaults = {key for key, row in zip(keys, cells) if "row" not in row}
    if len(keys) != len(set(keys)) or defaults != expected:
        raise ValueError("result does not account for every requested cell exactly once")
    if any((row["regime"], row["row"]) not in expected or not isinstance(row.get("alternative"), dict)
           for row in cells if "row" in row):
        raise ValueError("an alternative realization must name a requested row and its params")
    if any(row.get("state") not in {"RAN", "ERROR", "UNSUPPORTED"} for row in cells):
        raise ValueError("invalid execution cell status")
    if any(row["state"] == "UNSUPPORTED" and not (isinstance(row.get("unsupported"), str)
                                                  and row["unsupported"].strip()) for row in cells):
        raise ValueError("an UNSUPPORTED cell must name the missing capability")
    if file_digest(directory / "result.pt") != result.get("outputs_sha256"):
        raise ValueError("missing or corrupted tensor artifact")
    return result

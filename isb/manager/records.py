"""Read saved job reports as lightweight page records. Never open a tensor artifact."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ..jobs.contract import PLAN_VERSION, VERSIONS, check_experiment, expected_cells, wire_spec
from ..protocol import InterventionSpec
from ..runs import coordinate_config, run_coordinates, read_coordinates


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path.name}")
    value = json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def child(root, name):
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("invalid manifest path component")
    path = root / name
    if path.is_symlink() or path.resolve().parent != root.resolve():
        raise ValueError("manifest path escapes its directory")
    return path


def cells(rows):
    if not isinstance(rows, list):
        raise ValueError("cells must be a list")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("state"), str):
            raise ValueError("cell must have a string state")
        if not all(isinstance(row.get(k, "job"), str) for k in ("label", "workload")):
            raise ValueError("invalid cell identity")
        metrics = row.get("metrics") or {}
        if not isinstance(metrics, dict) or any(not isinstance(v, (int, float)) for v in metrics.values()):
            raise ValueError("invalid cell metrics")
        if any(row.get(k) is not None and not isinstance(row[k], (int, float))
               for k in ("median_latency_ms", "overhead_vs_baseline", "throughput", "peak_mem_mb")):
            raise ValueError("invalid performance measurement")
    return [SimpleNamespace(label=r.get("label", "job"), workload=r.get("workload", "job"),
                            state=r["state"], error=r.get("error"), metrics=r.get("metrics") or {},
                            latency_s=(r["median_latency_ms"] / 1000
                                       if r.get("median_latency_ms") is not None else None))
            for r in rows]


def auxiliary_calls(rows):
    """Optional supporting-call metadata supplied by workers; task statuses stay separate."""
    if not isinstance(rows, list):
        raise ValueError("auxiliary_calls must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("auxiliary call must be an object")
        key = row.get("key")
        if (not isinstance(key, list) or len(key) != 2 or not all(isinstance(k, str) for k in key)
                or key[0] not in {"__baseline__", "__effect__", "batched_perprompt"}
                or not isinstance(row.get("record"), dict)):
            raise ValueError("invalid auxiliary call key or record")
        if tuple(key) in seen:
            raise ValueError("duplicate auxiliary call")
        seen.add(tuple(key))
    return copy.deepcopy(rows)


def bundle_paths(root):
    """Find run bundles below a collection; do not descend inside runs, trash or symlinks."""
    root = Path(root)
    if not root.is_dir():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "experiments"
                         and not (Path(directory) / d).is_symlink())
        if "plan.json" in files:
            dirs[:] = []
            yield Path(directory)


def load_bundle(directory, prefix):
    """Return (entries, results, baselines, warnings) for a single recorded invocation."""
    entries, results, baselines, warnings = {}, {}, {}, []
    plan = read_json(directory / "plan.json")
    if plan.get("version") != PLAN_VERSION:
        raise ValueError("unknown plan version")
    names = plan["backends"]
    experiments = plan["experiments"]
    if not isinstance(names, list) or not isinstance(experiments, list):
        raise ValueError("backend and experiment identities must be lists")
    for name in names + experiments:
        child(directory, name)
    if not names or len(set(names)) != len(names) or len(set(experiments)) != len(experiments):
        raise ValueError("duplicate/empty manifest identities")
    reference = plan.get("reference")
    if reference is not None and reference not in names:
        raise ValueError("reference is not a selected backend")
    report = {}
    if (directory / "report.json").exists():
        try:
            saved = read_json(directory / "report.json")
            if saved.get("version") != PLAN_VERSION or saved.get("run") != directory.name:
                raise ValueError("report belongs to a different run/version")
            for item in saved["experiments"]:
                if not isinstance(item, dict):
                    raise ValueError("invalid report experiment")
                if item["experiment_id"] in report:
                    raise ValueError("duplicate experiment in report")
                report[item["experiment_id"]] = item
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {}
            warnings.append(f"{prefix}: report unavailable: {error}")
    for experiment_name in experiments:
        expdir = child(child(directory, "experiments"), experiment_name)
        try:
            experiment = read_json(expdir / "experiment.json")
            check_experiment(experiment)
            spec = wire_spec(experiment)
            expected = expected_cells(experiment)
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f"{prefix}/{experiment_name}: invalid experiment: {error}")
            continue
        group = f"{prefix}/{experiment_name}"
        scored = report.get(experiment["id"])
        scored_rows = {}
        if scored is not None:
            try:
                if scored.get("reference") != reference or scored.get("comparison") != plan.get("comparison"):
                    raise ValueError("report comparison differs from plan")
                cells(scored["cells"])
                for row in scored["cells"]:
                    key = (row["backend"], row.get("workload", "job"), row.get("label", "job"))
                    if (row["backend"] not in names or key in scored_rows
                            or key[1:] not in expected | {("job", "job")}):
                        raise ValueError("duplicate or unknown backend/cell in report")
                    scored_rows[key] = row
                for backend in names:
                    reported = {key[1:] for key in scored_rows if key[0] == backend}
                    if reported not in (expected, {("job", "job")}):
                        raise ValueError("report does not cover every requested cell")
            except (ValueError, KeyError, TypeError) as error:
                warnings.append(f"{group}: invalid report: {error}")
                scored_rows = {}
        for backend in names:
            key = f"{group}/{backend}"
            jobdir = child(expdir, backend)
            prov, rows, execution, auxiliary = {}, [], {}, []
            state = "PENDING" if plan.get("status") == "running" else "NOT_RUN"
            if plan.get("status") == "cancelled":
                state = "CANCELLED"
            error = None
            try:
                if (jobdir / "execution.json").exists():
                    execution = read_json(jobdir / "execution.json")
                    if execution.get("backend") != backend or execution.get("experiment_id") != experiment["id"]:
                        raise ValueError("execution identity mismatch")
                    state = {"completed": "COMPLETED", "running": "RUNNING", "cancelled": "CANCELLED"}.get(
                        execution.get("status"), "JOB_FAILED")
                    error = execution.get("error") or execution.get("cleanup_error")
                if (jobdir / "result.json").exists():
                    result = read_json(jobdir / "result.json")
                    if (type(result.get("version")) is not int or result["version"] not in VERSIONS or result.get("backend") != backend
                            or result.get("experiment_id") != experiment["id"]
                            or result.get("inputs_sha256") != experiment["inputs_sha256"]):
                        raise ValueError("result identity mismatch")
                    prov = copy.deepcopy(result.get("provenance", {}))
                    if not isinstance(prov, dict) or any(not isinstance(prov.get(k, {}), dict)
                                                        for k in ("client", "engine", "host", "deployment", "coordinates")):
                        raise ValueError("malformed provenance")
                    worker_coordinates = prov.get("coordinates", {})
                    if worker_coordinates.get("schema") is not None or "spec" in worker_coordinates:
                        prov["coordinates"] = read_coordinates(worker_coordinates)
                    auxiliary = auxiliary_calls(result.get("auxiliary_calls", []))
                    if auxiliary:
                        prov["auxiliary_calls"] = copy.deepcopy(auxiliary)
                    if state == "COMPLETED":
                        if result.get("status") != "completed" or not child(jobdir, "result.pt").is_file():
                            raise ValueError("completed job has no complete artifact")
                        raw = result["cells"]
                        cells(raw)
                        raw_keys = [(r["workload"], r["label"]) for r in raw]
                        if set(raw_keys) != expected or len(raw_keys) != len(expected):
                            raise ValueError("result does not cover requested cells")
                        rows = []
                        for item in raw:
                            saved = scored_rows.get((backend, item["workload"], item["label"]))
                            # Never turn an execution error into a passing verdict.
                            row = saved if saved and (item["state"] == "RAN" or saved["state"] == item["state"]) else item
                            rows.append({**item, **row})
                        job_failure = scored_rows.get((backend, "job", "job"))
                        if job_failure is not None:
                            rows = [{"workload": w, "label": label, "state": job_failure["state"],
                                     "error": job_failure.get("error")} for w, label in sorted(expected)]
                        if not scored_rows:
                            warnings.append(f"{key}: no saved comparison; displaying execution states only")
                elif state == "COMPLETED":
                    raise ValueError("completed job is missing result.json")
            except (OSError, ValueError, KeyError, TypeError) as caught:
                state, error, rows, prov, auxiliary = "JOB_FAILED", str(caught), [], {}, []
            if not rows:
                rows = [{"workload": w, "label": label, "state": state, "error": error}
                        for w, label in sorted(expected)]
            prov.setdefault("client", {})
            prov.setdefault("engine", {"kind": "unknown", "params": {}})
            prov["engine"].setdefault("kind", "unknown")
            prov.setdefault("deployment", {})
            prov.setdefault("host", {})
            # Submitted identity and the worker's executed description have distinct owners.
            # Preserve complete worker coordinates, including extension fields and call records.
            worker = prov.get("coordinates") or {}
            submitted = run_coordinates(
                spec=spec["name"], methodology=spec["methodology"], family=spec["family"],
                repo=spec["repo"], interface=worker.get("interface", "unknown"),
                regimes=[{**w, "data": w.get("data_name")} for w in spec["regimes"]],
                cases=spec["tasks"], protocol=(InterventionSpec(**spec["protocol"]).coordinate()
                                              if spec.get("protocol") is not None else None),
                protocol_coverage=({"status": "described"} if spec.get("protocol") is not None
                                   else {"status": "undescribed",
                                         "reason": spec.get("protocol_absence_reason")}),
                inputs_sha256=experiment["inputs_sha256"], config=coordinate_config(spec))
            prov["submitted_coordinates"] = submitted
            prov["coordinates"] = worker if "spec" in worker else {**submitted, **worker}
            prov["view"] = {"group": group, "backend": backend, "run": prefix,
                "experiment_id": experiment["id"], "inputs_sha256": experiment["inputs_sha256"],
                "reference": f"{group}/{reference}" if reference else None,
                "comparison": plan.get("comparison"), "job_status": state,
                "image_id": execution.get("image_id"), "source": execution.get("source"),
                "path": str(jobdir), "saved_report": bool(scored_rows)}
            prov.setdefault("executed", directory.name)
            meta = {tuple(call["key"]): call["record"] for call in auxiliary}
            meta.update({(r["workload"], r["label"]): r for r in rows})
            entries[key] = ({("__meta__",): meta}, prov)
            results[key] = cells(rows)
        if reference:
            baselines[group] = f"{group}/{reference}"
    return entries, results, baselines, warnings

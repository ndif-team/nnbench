"""Versions of the job artifacts, the fields of their timed records, and the upgrade of an older
artifact's rows, shared by the writers (launcher, worker, scorer) and every reader.

experiment.json and result.json carry the wire version, which also fixes the form of the
inputs.jsonl lines experiment.json binds by digest; plan.json, report.json and the coordinates in a
job's provenance carry their own. A row of result.json or report.json `cells`
names its regime in `regime` and its case in `label`. A timed record (a row, or the record of an
auxiliary call) carries `median_latency_ms` and `latency_kind`, what that median is of (the regime
kind's `latency`, isb/sweep/spec.py); an open-loop record also carries `saturated`, true when the
system did not keep up with the arrivals (isb/perf/load.py `kept_up`), so its latency grows with the
pass length. No latency ratio is formed from a saturated record or across latency kinds.

Wire version 2 and report version 1 named the regime `workload` and recorded neither field;
`upgrade_rows` and `upgrade_calls` add both and drop a saturated record's ratios, and
`drop_saturated_floors` drops a row's ratio over a saturated floor of its regime.
"""
from __future__ import annotations

WIRE_VERSION = 3          # experiment.json (with the inputs.jsonl it binds), result.json
WIRE_READ = (2, 3)
PLAN_VERSION = 1          # plan.json
REPORT_VERSION = 2        # report.json
REPORT_READ = (1, 2)
COORDINATES_VERSION = 4   # coordinates in a run's provenance; version 3 also held dtype_control
COORDINATES_READ = (3, 4)

RATIOS = ("overhead_vs_baseline", "overhead_vs_vanilla", "overhead_vs_plain_vllm",
          "overhead_vs_plain_vllm_default")
FLOORS = {"__baseline__": "overhead_vs_baseline", "__vanilla__": "overhead_vs_vanilla"}


def require(version, accepted, what):
    if type(version) is not int or version not in accepted:
        raise ValueError(f"unsupported {what} version {version!r}; rerun with current code")


def measured(record, regime):
    """Set a timed record's `latency_kind` from its regime key and, for an open-loop record, its
    `saturated` from the load summary. Returns the record."""
    from isb.perf.load import kept_up
    from isb.sweep.spec import regime_kind

    record["latency_kind"] = regime_kind(regime).latency
    load = record.get("load") or {}
    if load.get("offered_rate") is not None:
        record["saturated"] = not kept_up(load)
    return record


def latency_ratio(record, floor):
    """`record`'s median latency over `floor`'s, or None when either has none or is saturated.
    Raises on records whose latencies are of different kinds."""
    if (not floor or not floor.get("median_latency_ms") or record.get("median_latency_ms") is None
            or record.get("saturated") or floor.get("saturated")):
        return None
    if record["latency_kind"] != floor["latency_kind"]:
        raise ValueError(f"a {record['latency_kind']} latency over a {floor['latency_kind']} latency")
    return record["median_latency_ms"] / floor["median_latency_ms"]


def _upgrade(record, regime):
    if record.get("median_latency_ms") is None or regime is None:
        return record
    measured(record, regime)
    if record.get("saturated"):
        for key in RATIOS:
            record.pop(key, None)
    return record


def upgrade_rows(rows, version, current):
    """Cells of an artifact written at `version` (result.json: wire version, report.json: report
    version, `current` the version this code writes), in the current fields."""
    if version >= current:
        return rows
    upgraded = []
    for row in rows:
        row = dict(row)
        if "workload" in row:
            row["regime"] = row.pop("workload")
        upgraded.append(_upgrade(row, row.get("regime")))
    return upgraded


def drop_saturated_floors(rows, calls):
    """`rows` without the ratio over each regime floor (`calls` holding the `__baseline__` and
    `__vanilla__` records, upgraded) that is saturated; the worker forms none since wire version 3."""
    saturated = {tuple(call["key"]) for call in calls if call["record"].get("saturated")}
    for row in rows:
        for role, ratio in FLOORS.items():
            if (role, row.get("regime")) in saturated:
                row.pop(ratio, None)
    return rows


def upgrade_calls(calls, version):
    """result.json `auxiliary_calls` written at wire `version`, in the current fields. A call's key
    is [role, regime] (the per-unit stacks: [role, label], which hold no timing)."""
    if version >= WIRE_VERSION:
        return calls
    return [{**call, "record": _upgrade(dict(call["record"]), call["key"][1])} for call in calls]

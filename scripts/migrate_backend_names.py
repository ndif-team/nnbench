"""Rename saved runs from the flat backend names to `<system>/<config>` (backends/README.md).

    python scripts/migrate_backend_names.py runs/            # dry run: print what would change
    python scripts/migrate_backend_names.py runs/ --apply    # rewrite in place
    python scripts/migrate_backend_names.py runs/ --apply --reverse   # undo

Every run directory (one with plan.json) under the given roots is rewritten consistently: each job
directory moves to `experiments/<id>/<system>/<config>/`, and the name is replaced in plan.json
(`backends`, `reference`), execution.json and result.json (`backend`, `provenance.job.backend`),
result.pt (`provenance.job.backend`, with result.json's `outputs_sha256` updated to match), and
report.json (`reference` and each row's `backend`, `plain_vllm`, `plain_vllm_default`). Logs are
left as recorded. A run whose names are already current is skipped, so the script is idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAP = {
    "nnsight-hf": "nnsight-hf/default", "nnsight-hf-fp32": "nnsight-hf/fp32",
    "nnsight-hf-0-7": "nnsight-hf/0-7",
    "nnsight-vllm": "nnsight-vllm/default", "nnsight-vllm-fp32": "nnsight-vllm/fp32",
    "nnsight-vllm-0-7": "nnsight-vllm/0-7", "nnsight-vllm-sync": "nnsight-vllm/sync",
    "nnsight-vllm-opt": "nnsight-vllm/opt", "nnsight-vllm-taps": "nnsight-vllm/opt/taps",
    "transformer-lens": "transformer-lens-vllm/default",
    "transformer-lens-fp32": "transformer-lens-vllm/fp32",
    "transformer-lens-opt": "transformer-lens-vllm/opt",
    "vllm-lens": "vllm-lens/default", "vllm-lens-opt": "vllm-lens/opt",
    "vllm-hook": "vllm-hook/default", "vllm-hook-disk": "vllm-hook/disk",
    "interp-engine": "interp-engine-vllm/default", "interp-engine-static": "interp-engine-vllm/static",
    **{f"vllm-plain-{v}": f"vllm-plain/{v}" for v in ("0-15-1", "0-19-1", "0-20-2", "0-28-0")},
    **{f"vllm-plain-eager-{v}": f"vllm-plain/eager-{v}" for v in ("0-15-1", "0-19-1", "0-20-2", "0-28-0")},
    # removed configs (prefix caching off, d0fd79b), still named in saved diagnostic runs
    "nnsight-vllm-nopc": "nnsight-vllm/nopc", "nnsight-vllm-0-7-nopc": "nnsight-vllm/0-7-nopc",
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    """Atomic, in the writer's format (contract.write_json: sorted keys, compact)."""
    from isb.jobs.contract import write_json
    write_json(path, value)


def rename_tensor_artifact(job, mapping):
    """Rewrite result.pt's job identity; returns its new sha256, or None if unchanged."""
    import torch
    from isb.jobs.contract import file_digest

    path = job / "result.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    identity = artifact.get("provenance", {}).get("job", {})
    if identity.get("backend") not in mapping:
        return None
    identity["backend"] = mapping[identity["backend"]]
    partial = path.with_name(".result.pt.partial")
    torch.save(artifact, partial)
    os.replace(partial, path)
    return file_digest(path)


def migrate_job(job, mapping, apply, tensors):
    changes = []
    execution = job / "execution.json"
    if execution.is_file():
        record = read(execution)
        if record.get("backend") in mapping:
            record["backend"] = mapping[record["backend"]]
            changes.append("execution.json")
            if apply:
                write(execution, record)
    result_path = job / "result.json"
    if result_path.is_file():
        result = read(result_path)
        dirty = False
        if result.get("backend") in mapping:
            result["backend"] = mapping[result["backend"]]
            dirty = True
        identity = (result.get("provenance") or {}).get("job") or {}
        if identity.get("backend") in mapping:
            identity["backend"] = mapping[identity["backend"]]
            dirty = True
        if tensors and (job / "result.pt").is_file():
            changes.append("result.pt")
            if apply:
                digest = rename_tensor_artifact(job, mapping)
                if digest is not None:
                    result["outputs_sha256"] = digest
                    dirty = True
        if dirty:
            changes.append("result.json")
            if apply:
                write(result_path, result)
    return changes


def move_jobs(expdir, names, mapping, apply):
    """Move each job directory to its new path; parents (opt) are placed before children (opt/taps)."""
    staged = []
    for old in sorted(names, key=lambda name: -name.count("/")):    # children leave parents first
        source = expdir / old
        if not source.is_dir():
            continue
        temporary = expdir / (".migrate-" + old.replace("/", "__"))
        if apply:
            source.rename(temporary)
        staged.append((temporary, mapping[old]))
    if apply:                                         # parents left empty by nested moves (reverse)
        for old in names:
            parent = (expdir / old).parent
            while parent != expdir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    for temporary, new in sorted(staged, key=lambda item: item[1].count("/")):
        target = expdir / new
        if not apply:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():                                   # created as a nested config's parent
            for item in temporary.iterdir():
                item.rename(target / item.name)
            temporary.rmdir()
        else:
            temporary.rename(target)
    return [new for _, new in staged]


def rename_report(report, mapping):
    for experiment in report.get("experiments", []):
        if experiment.get("reference") in mapping:
            experiment["reference"] = mapping[experiment["reference"]]
        for row in experiment.get("cells", []):
            for key in ("backend", "plain_vllm", "plain_vllm_default"):
                if row.get(key) in mapping:
                    row[key] = mapping[row[key]]


def migrate_run(run, mapping, apply, tensors):
    plan = read(run / "plan.json")
    names = plan.get("backends", [])
    unknown = [n for n in names if n not in mapping and n not in mapping.values()]
    if unknown:
        return f"skipped: unknown backend names {unknown}"
    old = [n for n in names if n in mapping]
    if not old:
        return None
    files = 0
    for experiment in plan.get("experiments", []):
        expdir = run / "experiments" / experiment
        moved = move_jobs(expdir, old, mapping, apply)
        for new in moved:
            job = expdir / new if apply else expdir / next(o for o in old if mapping[o] == new)
            files += len(migrate_job(job, mapping, apply, tensors))
    plan["backends"] = [mapping.get(n, n) for n in names]
    if plan.get("reference") in mapping:
        plan["reference"] = mapping[plan["reference"]]
    if apply:
        write(run / "plan.json", plan)
    if (run / "report.json").is_file():
        report = read(run / "report.json")
        rename_report(report, mapping)
        if apply:
            write(run / "report.json", report)
    return f"{len(old)} backends, {files} job files"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true", help="rewrite in place (default: dry run)")
    ap.add_argument("--reverse", action="store_true", help="map the new names back to the old ones")
    ap.add_argument("--no-tensors", action="store_true", help="leave result.pt files untouched")
    args = ap.parse_args(argv)
    mapping = {v: k for k, v in MAP.items()} if args.reverse else MAP
    for root in args.roots:
        for plan in sorted(root.rglob("plan.json")):
            if ".migrate-" in str(plan):
                continue
            outcome = migrate_run(plan.parent, mapping, args.apply, not args.no_tensors)
            if outcome and outcome.startswith("skipped"):
                print(f"{plan.parent}: {outcome}", flush=True)
            elif outcome:
                print(f"{'migrated' if args.apply else 'would migrate'} {plan.parent}: {outcome}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

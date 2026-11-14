#!/usr/bin/env python3
"""Audit the run directories of an evaluation before its numbers are used.

Usage:
    python3 scripts/audit_runs.py RUNS/cross-system
    python3 scripts/audit_runs.py --allow-commits d0ed4b9,1a2b3c4 RUNS

Every run dir must come from one (commit, source_sha256): score.py pairs a candidate with its reference
only when both ran from the same source. Across run dirs one commit is required unless --allow-commits
names the commits (prefixes accepted) the runs may span; each commit must still map to one source_sha256.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isb.jobs.schema import REPORT_VERSION, upgrade_rows  # noqa: E402


def audit_run_dir(run_dir: Path) -> tuple[list[str], list[str], dict]:
    errors, warnings = [], []
    plan_file = run_dir / "plan.json"
    report_file = run_dir / "report.json"

    if not plan_file.exists():
        return [f"{run_dir}: missing plan.json"], [], {}
    if not report_file.exists():
        return [f"{run_dir}: missing report.json"], [], {}

    plan = json.loads(plan_file.read_text())
    report = json.loads(report_file.read_text())

    if plan.get("status") != "completed":
        errors.append(f"{run_dir.name}: plan.status is {plan.get('status')!r}, expected 'completed'")

    has_plain_vllm = any(b.startswith("vllm-plain/") for b in plan.get("backends", []))

    sources = set()   # (commit, source_sha256) per job
    exec_files = sorted(run_dir.glob("experiments/*/**/execution.json"))
    if not exec_files:
        errors.append(f"{run_dir.name}: no execution.json files found")

    for ef in exec_files:
        ex = json.loads(ef.read_text())
        rel = ef.relative_to(run_dir)
        if ex.get("status") != "completed" or ex.get("exit_code") != 0:
            errors.append(f"{rel}: status={ex.get('status')!r}, exit_code={ex.get('exit_code')}")
        src = ex.get("source") or {}
        sources.add((src.get("commit"), src.get("source_sha256")))
        if not ex.get("image_id"):
            errors.append(f"{rel}: missing image_id")

        # The launcher refuses a job whose reserved GPU is busy; it can only skip the check when
        # nvidia-smi cannot report. Jobs launched before that check record `gpu_release`.
        before = ex.get("gpus_before")
        if before is None:
            release = ex.get("gpu_release") or {}
            base_mib, after_mib = release.get("baseline_mib"), release.get("after_mib")
            if base_mib is None or base_mib >= 100:
                errors.append(f"{rel}: dirty GPU pre-flight baseline_mib={base_mib} (must be < 100 MiB)")
            elif after_mib is not None and after_mib > base_mib + 1024:
                warnings.append(f"{rel}: GPU memory did not fully drain (after_mib={after_mib})")
        elif not before.get("state"):
            errors.append(f"{rel}: the reserved GPUs were not checked before the job")
    if len(sources) > 1:
        errors.append(f"{run_dir.name}: SOURCE DRIFT within the run: {sorted(sources, key=str)}")

    # Check model_identity (repo, revision, vocab_sha256) consistency per experiment
    for exp_dir in sorted((run_dir / "experiments").glob("*")):
        identities = set()
        for rf in exp_dir.glob("**/result.json"):
            res = json.loads(rf.read_text())
            mi = (res.get("provenance") or {}).get("model_identity") or {}
            if mi:
                identities.add((mi.get("repo"), mi.get("revision"), mi.get("vocab_sha256")))
        if len(identities) > 1:
            errors.append(
                f"{run_dir.name}/{exp_dir.name}: mismatched model_identity across backends: {identities}"
            )

    # Check cells in report.json
    state_counts = {}
    high_cv_cells = 0
    for exp in report.get("experiments", []):
        spec_name = exp.get("spec")
        for cell in upgrade_rows(exp.get("cells", []), report.get("version", 1), REPORT_VERSION):
            st = cell.get("state", "UNKNOWN")
            backend = cell.get("backend", "")
            state_counts[st] = state_counts.get(st, 0) + 1
            loc = f"{run_dir.name}:{spec_name}:{backend}:{cell.get('label')}"
            if st in ("JOB_FAILED", "NO_REFERENCE", "INCOMPATIBLE", "INVALID_OUTPUT"):
                errors.append(f"{loc} -> hard failure state {st} ({cell.get('error')})")
            elif st == "INVALID_REFERENCE":
                warnings.append(f"{loc} -> INVALID_REFERENCE (weak reference effect: {cell.get('error')})")
            if st == "UNSUPPORTED" and not cell.get("unsupported") and not backend.startswith("vllm-plain"):
                warnings.append(f"{loc} -> UNSUPPORTED without reason string")

            med = cell.get("median_latency_ms")
            std = cell.get("std_latency_ms")
            if med and std and med > 0 and (std / med) > 0.20:
                high_cv_cells += 1
                warnings.append(
                    f"{loc} -> high latency variance CV={std/med:.2f} (median={med:.1f}ms, std={std:.1f}ms)"
                )

            # Check overhead_vs_vanilla and overhead_vs_plain_vllm on timed vLLM rows; a
            # saturated open-loop row has none by design
            if med and not cell.get("saturated") and not backend.startswith(("nnsight-hf/", "vllm-plain/")):
                if cell.get("overhead_vs_vanilla") is None:
                    warnings.append(f"{loc} -> timed vLLM row missing overhead_vs_vanilla")
                if has_plain_vllm and cell.get("overhead_vs_plain_vllm") is None:
                    errors.append(f"{loc} -> timed vLLM row missing overhead_vs_plain_vllm (check matching vllm-plain version/eager config)")

    summary = {
        "run": run_dir.name,
        "jobs": len(exec_files),
        "sources": sorted(s for s in sources if all(s)),
        "states": state_counts,
        "high_cv_cells": high_cv_cells,
    }
    return errors, warnings, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--allow-commits", default="", metavar="C1,C2",
                    help="commits (prefixes accepted) the run dirs may span")
    args = ap.parse_args()
    allowed = [c for c in args.allow_commits.split(",") if c]
    run_dirs = []
    for r in args.roots:
        if (r / "plan.json").exists():
            run_dirs.append(r)
        else:
            run_dirs.extend(sorted(p.parent for p in r.glob("**/plan.json")))

    if not run_dirs:
        print("No run directories with plan.json found.", file=sys.stderr)
        sys.exit(1)

    all_errors, all_warnings = [], []
    hashes = {}   # commit -> the source_sha256 values it ran with, over every run dir

    for rd in run_dirs:
        errs, warns, info = audit_run_dir(rd)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        for commit, sha in info.get("sources", []):
            hashes.setdefault(commit, set()).add(sha)
        print(
            f"[AUDIT] {info.get('run', rd.name)}: jobs={info.get('jobs', 0)} "
            f"states={info.get('states', {})} high_cv={info.get('high_cv_cells', 0)}"
        )

    for commit, shas in sorted(hashes.items()):
        if len(shas) > 1:
            all_errors.append(f"GLOBAL SOURCE DRIFT: commit {commit} ran with several source_sha256: {sorted(shas)}")
    stray = sorted(c for c in hashes if not any(c.startswith(a) for a in allowed))
    if allowed and stray:
        all_errors.append(f"GLOBAL SOURCE DRIFT: commits outside --allow-commits: {stray}")
    if not allowed and len(hashes) > 1:
        all_errors.append(f"GLOBAL SOURCE DRIFT: commits={sorted(hashes)} (--allow-commits accepts several)")

    for w in all_warnings[:25]:
        print(f"  [WARN] {w}")
    for e in all_errors:
        print(f"  [FAIL] {e}")

    if all_errors:
        print(f"\nAUDIT FAILED: {len(all_errors)} error(s), {len(all_warnings)} warning(s).")
        sys.exit(1)
    print(f"\nAUDIT PASSED: {len(run_dirs)} run(s) verified with 0 errors ({len(all_warnings)} warnings).")


if __name__ == "__main__":
    main()

"""Build and run independent backend directories using a small, name-based CLI."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .contract import PLAN_VERSION, prepare, write_json
from . import apptainer
from .local import (LAUNCHERS, ROOT, backend_file, canonical, compose, configuration, default_launcher,
                    discover, environment, project_name, run_job)


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _specs(names, data):
    from isb.data import DataRef
    from isb.specs import SPECS, default_specs
    from isb.sweep.spec import spec_with_data

    selected = default_specs() if names == ["all"] else names
    if "all" in selected or len(selected) != len(set(selected)):
        raise ValueError("select unique spec names, or 'all' alone")
    ref = None
    if data:
        name, separator, count = data.partition(":")
        if separator and (not count.isdigit() or int(count) < 1):
            raise ValueError("dataset count must be positive")
        ref = DataRef(name, int(count) if separator else None)
    results = []
    for name in selected:
        if name not in SPECS:
            raise ValueError(f"unknown spec {name!r}; available: {', '.join(SPECS)}")
        results.append(spec_with_data(SPECS[name], ref) if ref else SPECS[name])
    return results


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("kind", choices=["backends", "specs"])
    build = commands.add_parser("build")
    build.add_argument("backends", nargs="+")
    build.add_argument("--launcher", choices=LAUNCHERS, default=None,
                       help="docker or apptainer (default: ISB_LAUNCHER, else docker)")
    run = commands.add_parser("run")
    run.add_argument("--spec", nargs="+", required=True)
    run.add_argument("--data")
    run.add_argument("--backends", nargs="+", required=True,
                     help="config names, <system>/<config>; a bare system name means <system>/default")
    run.add_argument("--reference", help="one of --backends; omit to collect without scoring")
    run.add_argument("--comparison", choices=["correctness", "equivalence"], default="correctness")
    run.add_argument("--gpu", default="0", help="host GPU index or UUID; passed to backend Compose")
    run.add_argument("--timeout", type=_positive, default=1800, help="seconds per backend job")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--launcher", choices=LAUNCHERS, default=None,
                     help="docker or apptainer (default: ISB_LAUNCHER, else docker)")
    run.add_argument("--out", type=Path, default=Path("runs"), help="parent of a new unique run directory")
    run.add_argument("--strict", action="store_true", help="nonzero exit for failed execution/output checks; numerical uncertainty is nonblocking")
    scoring = commands.add_parser("score")
    scoring.add_argument("run", type=Path)
    scoring.add_argument("--strict", action="store_true")
    return ap


def _score(directory, strict):
    # Keep tensor imports out of the Docker launcher. CPU torch is only needed by this child.
    result = subprocess.run([sys.executable, "-m", "isb.jobs.cli", "score", str(directory),
                             *(["--strict"] if strict else [])], cwd=ROOT)
    return result.returncode


def _run(args):
    # Records carry only full config names; a bare system name means its default config.
    args.backends = [canonical(name) for name in args.backends]
    args.reference = canonical(args.reference) if args.reference else None
    if len(args.backends) != len(set(args.backends)):
        raise ValueError("backend names must be unique")
    if args.reference and args.reference not in args.backends:
        raise ValueError("reference must be one of --backends")
    if not args.gpu or "," in args.gpu:
        raise ValueError("select one GPU index/UUID; multi-GPU settings belong in backend Compose")
    specs = _specs(args.spec, args.data)
    launcher = args.launcher or default_launcher()
    for name in args.backends:
        env = environment(name, args.out, args.out, args.gpu)
        if launcher == "apptainer":
            apptainer.load_service(backend_file(name), env)
        else:
            configuration(name, env=env)
    # Check scorer availability before spending GPU time, without importing it into the launcher.
    subprocess.run([sys.executable, "-c", "import torch"], check=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.out.resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    plan = {"version": PLAN_VERSION, "backends": args.backends, "reference": args.reference,
            "comparison": args.comparison, "gpu": args.gpu, "launcher": launcher, "experiments": [],
            "status": "preparing"}
    jobs = []
    for index, spec in enumerate(specs):
        # Preparation uses an index until the content identity is known, then renames atomically.
        pending = directory / "experiments" / f"prepare-{index}"
        experiment = prepare(spec, pending, seed=args.seed)
        target = pending.with_name(experiment["id"][:16])
        pending.rename(target)
        plan["experiments"].append(target.name)
        jobs.append((target, experiment))
    plan["status"] = "running"
    write_json(directory / "plan.json", plan)
    print(f"Run directory: {directory}", flush=True)
    failed = False
    try:
        for job, experiment in jobs:
            for name in args.backends:
                record = run_job(name, job, experiment, job / name, gpu=args.gpu, timeout=args.timeout,
                                 launcher=launcher)
                failed |= record["status"] != "completed"
        plan["status"] = "failed" if failed else "completed"
    except KeyboardInterrupt:
        plan["status"] = "cancelled"
        raise
    finally:
        write_json(directory / "plan.json", plan)
    score_code = _score(directory, args.strict)
    print(f"Report: {directory / 'report.json'}", flush=True)
    return 1 if failed else score_code


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "list":
            if args.kind == "backends":
                print("\n".join(discover()))
            else:
                from isb.specs import SPECS
                print("\n".join(sorted(SPECS)))
            return 0
        if args.command == "build":
            # Resolve all names before the first build; do not partially act on a typo.
            args.backends = [canonical(name) for name in args.backends]
            launcher = args.launcher or default_launcher()
            for name in args.backends:
                if launcher == "apptainer":
                    print(f"[build] {name} -> {apptainer.build(backend_file(name), ROOT, dict(os.environ))}",
                          flush=True)
                else:
                    configuration(name)
                    subprocess.run(compose(name, project_name("isb-build-", name)) + ["build"], check=True)
            return 0
        if args.command == "run":
            return _run(args)
        from .score import score_run
        report = score_run(args.run.resolve())
        states = {cell["state"] for exp in report["experiments"] for cell in exp["cells"]}
        if states & {"JOB_FAILED"}:
            return 1
        if args.strict and states - {"RAN", "SUPPORTED", "EQUIVALENT", "NUMERICAL_MISMATCH",
                                    "DIVERGENT", "NO_REFERENCE", "INVALID_REFERENCE", "INCOMPATIBLE",
                                    "SUPPORTED_DEGRADED", "EQUIVALENT_DEGRADED"}:
            return 1
        return 0
    except KeyboardInterrupt:
        print("Interrupted; job cleanup attempted. See execution.json.", file=sys.stderr)
        return 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"bench: {error}", file=sys.stderr)
        return 1


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())

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

from isb.suites import cases

from .contract import PLAN_VERSION, prepare, write_json
from . import apptainer
from .local import (LAUNCHERS, ROOT, backend_file, canonical, compose, configuration, default_launcher,
                    discover, environment, project_name, run_job)


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _recipe(args):
    """The suite (or ad-hoc combination) the run executes, with command-line overrides applied.
    Returns (suite name or None, entries, settings)."""
    from isb.suites import SUITES, Entry, parse_data, parse_load

    parse_data(args.data)                            # fail on a malformed source early
    loads = tuple(parse_load(kind, text) for kind in ("offline", "concurrent")
                  for text in getattr(args, kind) or ())

    if args.suite:
        if args.suite not in SUITES:
            raise ValueError(f"unknown suite {args.suite!r}; available: {', '.join(SUITES)}")
        suite = SUITES[args.suite]
        settings = {"backends": list(suite.backends), "reference": suite.reference,
                    "realizations": suite.realizations, "comparison": suite.comparison,
                    "timeout": suite.timeout}
        # --workload/--model with a suite keep only those of its cases
        entries = [Entry(tuple(w for w in e.workloads if not args.workload or w in args.workload),
                         tuple(m for m in e.models if not args.model or m in args.model),
                         size=e.size, data=e.data, regimes=e.regimes, loads=e.loads)
                   for e in suite.entries]
        entries = [e for e in entries if e.workloads and e.models]
        if not entries:
            raise ValueError(f"suite {args.suite!r} has no case among the selected workloads/models")
    else:
        if not args.workload or not args.model:
            raise ValueError("select --suite, or --workload and --model")
        settings = {"backends": None, "reference": None, "realizations": "all",
                    "comparison": "correctness", "timeout": 1800}
        entries = [Entry(tuple(args.workload), tuple(args.model))]
    for key in settings:
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if not settings["backends"]:
        raise ValueError("select --backends")
    if args.data or args.regimes or loads:
        entries = [Entry(e.workloads, e.models, size=e.size, data=args.data or e.data,
                         regimes=tuple(args.regimes) if args.regimes else e.regimes,
                         loads=loads or e.loads) for e in entries]
    return args.suite, entries, settings


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("kind", choices=["backends", "workloads", "models", "suites"])
    build = commands.add_parser("build")
    build.add_argument("backends", nargs="+")
    build.add_argument("--launcher", choices=LAUNCHERS, default=None,
                       help="docker or apptainer (default: ISB_LAUNCHER, else docker)")
    run = commands.add_parser("run", description="Run a suite (isb/suites.py), or workloads x models. "
                              "With --suite, other options override the suite's settings and "
                              "--workload/--model keep only those of its cases.")
    run.add_argument("--suite")
    run.add_argument("--workload", nargs="+", help="workload names (isb/workloads/)")
    run.add_argument("--model", nargs="+", help="model keys (isb/models.py)")
    run.add_argument("--data", help="SOURCE[:N]: rebind every regime to a registered data source")
    run.add_argument("--regimes", nargs="+",
                     choices=["interactive", "batched", "generation", "offline", "concurrent"],
                     help="keep only these regime kinds")
    run.add_argument("--offline", action="append", metavar="UNITS:B[,B...]",
                     help="add the offline load pattern (design.md §12.17): UNITS units of each "
                          "workload's batched data, B units per call (or 'all'); replaces the "
                          "suite's load patterns")
    run.add_argument("--concurrent", action="append", metavar="UNITS:C[,C...]",
                     help="add the closed-loop concurrent load pattern: UNITS single-unit requests "
                          "from C clients; replaces the suite's load patterns")
    run.add_argument("--realizations", choices=["default", "all"],
                     help="each system's documented realization only, or also every alternative its "
                          "cells declare (ad-hoc default: all)")
    run.add_argument("--backends", nargs="+",
                     help="config names, <system>/<config>; a bare system name means <system>/default")
    run.add_argument("--reference", help="one of --backends; omit (or '' with --suite) to collect "
                                         "without scoring")
    run.add_argument("--comparison", choices=["correctness", "equivalence"])
    run.add_argument("--gpu", default="0", help="host GPU index or UUID; passed to backend Compose")
    run.add_argument("--timeout", type=_positive, help="seconds per backend job (default 1800)")
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
    suite, entries, settings = _recipe(args)
    # Records carry only full config names; a bare system name means its default config.
    args.backends = [canonical(name) for name in settings["backends"]]
    args.reference = canonical(settings["reference"]) if settings["reference"] else None
    args.comparison, args.timeout = settings["comparison"], settings["timeout"]
    if len(args.backends) != len(set(args.backends)):
        raise ValueError("backend names must be unique")
    if args.reference and args.reference not in args.backends:
        raise ValueError("reference must be one of --backends")
    if not args.gpu or "," in args.gpu:
        raise ValueError("select one GPU index/UUID; multi-GPU settings belong in backend Compose")
    specs, skipped = cases(entries, settings["realizations"])
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
    plan = {"version": PLAN_VERSION, "suite": suite, "backends": args.backends,
            "reference": args.reference, "comparison": args.comparison,
            "realizations": settings["realizations"], "gpu": args.gpu, "launcher": launcher,
            "experiments": [], "cases": [spec.name for spec in specs], "not_applicable": skipped,
            "status": "preparing"}
    for row in skipped:
        print(f"not applicable: {row['case']} | {row['label']} | {row['reason']}", flush=True)
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
            elif args.kind == "workloads":
                from isb.workloads import WORKLOADS
                print("\n".join(WORKLOADS))
            elif args.kind == "models":
                from isb.models import MODELS
                print("\n".join(f"{m.key}\t{m.repo}\t{m.family}\t{m.n_layers} layers"
                                for m in MODELS.values()))
            else:
                from isb.suites import SUITES
                print("\n".join(SUITES))
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

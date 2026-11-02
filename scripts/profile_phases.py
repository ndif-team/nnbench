"""Phase-profile workloads on backend configs (isb/perf/phase_profiler.py), one container per
(case, config), through each system's own compose.yml with the profiler as the entrypoint.

    python3 scripts/profile_phases.py --backends nnsight-vllm/0-8-1 vllm-lens/default \
        --workload steering logit_lens gen_steering --model qwen2.5-7b --gpu 6 --out DIR

Writes DIR/<stamp>/<case>/<config>/phase_profile.json and execution.log, and prints each row's
raw segments and the conservation check.

    python3 scripts/profile_phases.py --combine DIR

splits every profile under DIR with its compiled vllm-plain floor of the same vLLM version and
GPU count (isb.perf.phase_profiler.split_with_floors; profile the floors into the same DIR) and
writes DIR/latency_breakdown-<attribution>.json (the event attribution by default,
`--attribution spans` for the span attribution).
"""
import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.jobs.contract import prepare  # noqa: E402
from isb.jobs.local import (canonical, compose, config, configuration,  # noqa: E402
                             environment, gpu_used_mib, wait_for_release)
from isb.perf.phase_profiler import ATTRIBUTIONS, split_with_floors  # noqa: E402
from isb.suites import Entry, cases  # noqa: E402


def run(name, job, output, gpu, timeout, rows, marks=True):
    output.mkdir(parents=True, exist_ok=True)
    project = "isb-phase-" + uuid.uuid4().hex[:16]
    command = compose(name, project)
    env = {**environment(name, job, output, gpu), "ISB_PHASE_ROWS": json.dumps(rows),
           "ISB_PHASE_MARKS": "1" if marks else "0"}
    configuration(name, env=env)
    baseline = gpu_used_mib(gpu)
    with (output / "execution.log").open("w") as log:
        try:
            code = subprocess.run(command + ["run", "--rm", "--no-TTY", "-e", "ISB_PHASE_PROFILE=1",
                                             "-e", "ISB_PHASE_ROWS", "-e", "ISB_PHASE_MARKS",
                                             "--entrypoint", "python3",
                                             "runner", "-m", "isb.perf.phase_profiler"],
                                  env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = f"timeout after {timeout} s"
        finally:
            subprocess.run(command + ["down", "--remove-orphans", "--timeout", "10"], env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=60)
    wait_for_release(gpu, baseline)
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backends", nargs="+")
    ap.add_argument("--combine", type=Path, help="split the profiles under DIR with their floors")
    ap.add_argument("--attribution", choices=sorted(ATTRIBUTIONS), default="events",
                    help="which raw segments --combine splits (default: events)")
    ap.add_argument("--workload", nargs="+", default=["steering", "logit_lens", "gen_steering"])
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--data", help="SOURCE[:N]: rebind every regime to a registered data source")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--rows", nargs="+", help="row labels to profile (default: every row)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--trace-only", action="store_true",
                    help="open the trace session but record no region boundaries: the tracing's "
                         "own cost")
    args = ap.parse_args()
    if args.combine:
        return combine(args.combine, args.attribution)
    if not (args.backends and args.out):
        ap.error("--backends and --out are required unless --combine is given")

    specs, _ = cases([Entry(tuple(args.workload), (args.model,), size=args.size, data=args.data,
                            regimes=("interactive", "generation"))])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = args.out.resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
    failed = False
    for spec in specs:
        job = directory / spec.name.replace("/", "-").replace("@", "-") / "job"
        prepare(spec, job)
        for name in [canonical(b) for b in args.backends]:
            output = job.parent / name.replace("/", "-")
            code = run(name, job, output, args.gpu, args.timeout, args.rows,
                       marks=not args.trace_only)
            profile = output / "phase_profile.json"
            if code != 0 or not profile.exists():
                failed = True
                print(f"[failed] {spec.name} | {name}: exit {code}; see {output / 'execution.log'}")
                continue
            for key, row in json.loads(profile.read_text())["rows"].items():
                if "error" in row:
                    print(f"{spec.name} | {name} | {key} | ERROR {row['error'][:200]}")
                    continue
                seg = " ".join(f"{k}={v:.2f}" for k, v in row["segments_raw_ms"].items())
                c = row["conservation"]
                print(f"{spec.name} | {name} | {key} | e2e={row['e2e_wall_ms']:.2f} "
                      f"uninstrumented={row['uninstrumented_e2e_ms']:.2f} {seg} | "
                      f"gap={c['max_abs_gap_ms']:.3f} ordered={c['all_ordered']} "
                      f"single={c['single_request']}")
    print(f"Run directory: {directory}")
    return 1 if failed else 0


def combine(directory, attribution):
    profiles = [json.loads(p.read_text()) for p in sorted(directory.rglob("phase_profile.json"))]
    for profile in profiles:          # the GPUs its config reserves: the engine's TP size
        profile["gpus"] = config(profile["backend"]).gpus
    records = split_with_floors(profiles, attribution)
    out = directory / f"latency_breakdown-{attribution.replace('_', '-')}.json"
    out.write_text(json.dumps(records, indent=1))
    for r in records:
        head = f"{r['case']} | {r['backend']} | {r['row']}"
        if "error" in r:
            print(f"{head} | ERROR {r['error'][:200]}")
            continue
        seg = " ".join(f"{k}={v:.2f}" for k, v in r["segments_ms"].items())
        c = r["conservation"]
        print(f"{head} | e2e={r['e2e_wall_ms']:.2f} "
              f"uninstrumented={r['uninstrumented_e2e_ms']:.2f} {seg} | floor={r['floor']} "
              f"F={r['floor_engine_ms']:.2f} | trial_gap={c['max_trial_gap_ms']:.2g} "
              f"ordered={c['all_ordered']} single={c['single_request']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

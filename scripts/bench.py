"""Single benchmark entrypoint — orchestrates execute/score runs (design.md §12.9, §12.10).

For each spec, every run executes in its OWN scripts/execute.py subprocess and writes one
self-contained run file (outputs + provenance) into --out; scoring is then a pure-CPU step over
those files (isb/sweep/score.py). The reference and the system under test never share a process,
an env, or CUDA state; `--backend-python` puts a backend's run under a different interpreter
(its own conda env). PP/TP equivalence mode (--pp/--tp > 1) splits the same way: the single-GPU
(1,1) control and the (tp,pp) candidate each get their own engine process, and the candidate is
scored against the control (same engine kind -> score.py picks the equivalence axis).

    CUDA_VISIBLE_DEVICES=4 python scripts/bench.py --spec steering_gpt2
    CUDA_VISIBLE_DEVICES=4 python scripts/bench.py --spec all --backends hf
    # control run in another env (arch newer than the vllm pin, e.g. qwen3_5):
    CUDA_VISIBLE_DEVICES=4 python scripts/bench.py --spec jacobian_lens_qwen35 \
        --backend-python hf=/disk/u/zikai/anaconda3/envs/nnsight-tf/bin/python
    # GPU-less serve client: execute over HTTP, score vs run files already in --out:
    python scripts/bench.py --spec all --backends vllm_serve --serve http://server:6677 \
        --out /refs --score-vs hf

Must be run under the `if __name__ == "__main__"` guard (vLLM EngineCore uses spawn).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.runfile import INBOX, run_path  # noqa: E402
from isb.specs import SPECS, default_specs  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402
from isb.sweep.split import (  # noqa: E402
    backend_run_commands,
    ctl_run_command,
    ctl_run_name,
    parallel_run_commands,
    run_base,
)


def _execute_runs(spec_name, rows, failed_runs):
    """Run one spec's execute.py subprocesses sequentially; returns the run names that completed.
    The first row is the reference — if it fails, the later runs have nothing to be scored
    against, so they are skipped."""
    done = []
    for i, (label, run_name, argv) in enumerate(rows):
        print(f"\n########## spec: {spec_name} — {label} run ##########", flush=True)
        rc = subprocess.run(argv).returncode
        if rc != 0:
            failed_runs.append((spec_name, label, rc))
            if i == 0:
                print(f"[bench] reference run failed (rc={rc}); "
                      f"skipping the runs scored against it")
                break
        else:
            done.append(run_name)
    return done


def _bound_spec(name, data):
    """The spec with its data binding applied (--data rebinds every workload)."""
    spec = SPECS[name]
    if data:
        from isb.data import DataRef
        from isb.sweep.spec import spec_with_data
        src, _, n = data.partition(":")
        spec = spec_with_data(spec, DataRef(src, int(n) if n else None))
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="all", help="spec name or 'all'")
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    ap.add_argument("--serve", default=None,
                    help="nnsight-vllm-serve URL for the vllm_serve backend (e.g. http://server:6677)")
    ap.add_argument("--out", default=INBOX, help="run-file directory (default: the inbox)")
    ap.add_argument("--no-ctl", action="store_true",
                    help="skip the control-dtype vLLM run; an undisambiguated precision "
                         "near-tie then stays SILENTLY_WRONG")
    ap.add_argument("--ctl-only", action="store_true",
                    help="execute ONLY the control-dtype vLLM run per spec (e.g. to stock a "
                         "serve client's reference directory); no scoring")
    ap.add_argument("--score-vs", default=None, metavar="TAG",
                    help="skip the reference run: score every executed run against the existing "
                         "run file <spec>-TAG in --out (the GPU-less serve client's mode)")
    # ---- parallelism-equivalence mode (PP/TP) --------------------------------------------------
    # PP>1 or TP>1 scores a parallel vLLM engine against single-GPU vLLM instead of against HF:
    # control = vLLM (1,1) [GT2: same intervention at tp=1,pp=1], candidate = vLLM (tp,pp). No HF
    # reference is run. This is the right correctness model for pipeline/tensor parallelism
    # (where bitwise equivalence to HF is neither expected nor the question).
    ap.add_argument("--pp", type=int, default=1,
                    help="pipeline_parallel_size (>1 enables PP/TP equivalence mode)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size (>1 enables PP/TP equivalence mode)")
    ap.add_argument("--executor", default=None, choices=["ray", "mp"],
                    help="vLLM distributed_executor_backend for the (tp,pp) candidate; 'ray' for multi-node")
    ap.add_argument("--gpu-mem", type=float, default=0.2,
                    help="gpu_memory_utilization for PP/TP mode (raise for large models; 0.2 suits gpt2)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="cap max_model_len in PP/TP mode (keeps KV small so big weights + KV fit one GPU)")
    ap.add_argument("--backend-python", action="append", default=[], metavar="BACKEND=PATH",
                    help="interpreter for one backend's run, e.g. hf=/path/envs/nnsight-tf/bin/python "
                         "(repeatable; default: this interpreter). Lets each backend run in its own env.")
    ap.add_argument("--release", action="store_true",
                    help="serious-measurement mode: every run process verifies its environment is "
                         "clean (visible GPUs free of other work, disk headroom) and REFUSES on "
                         "contamination, naming the offending processes. Never waits. Off = just run.")
    ap.add_argument("--data", default=None, metavar="SOURCE[:N]",
                    help="rebind every spec's workloads to a registered data source (isb/data.py), "
                         "e.g. jlens/poetry or factual:64 — same procedure, different data; runs "
                         "are named <spec>@<source>-* so run files never collide")
    args = ap.parse_args()

    parallel = args.pp > 1 or args.tp > 1
    python_map = dict(kv.split("=", 1) for kv in args.backend_python)
    execute = str(ROOT / "scripts" / "execute.py")
    names = default_specs() if args.spec == "all" else [args.spec]
    failed_runs = []

    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown spec {name!r}; choices: {', '.join(SPECS)} or 'all'")
        spec = _bound_spec(name, args.data)
        if args.ctl_only:
            rows = [("ctl", ctl_run_name(name, args.data, spec.dtype_control),
                     ctl_run_command(execute, name, args.out, spec.dtype_control,
                                     python=python_map.get("vllm_async"),
                                     data=args.data, release=args.release))]
        elif parallel:
            print(f"\n########## spec: {name} "
                  f"[PP={args.pp} TP={args.tp} executor={args.executor or 'mp'}] ##########")
            rows = parallel_run_commands(execute, name, args.out, pp=args.pp, tp=args.tp,
                                         executor=args.executor, gpu_mem=args.gpu_mem,
                                         max_model_len=args.max_model_len,
                                         dtype_control=spec.dtype_control)
        else:
            ctl_dtype = None if (args.no_ctl or args.score_vs or len(args.backends) < 2) \
                else spec.dtype_control
            rows = backend_run_commands(execute, name, args.backends, args.out,
                                        python_map=python_map, serve=args.serve,
                                        data=args.data, release=args.release,
                                        ctl_dtype=ctl_dtype)
        done = _execute_runs(name, rows, failed_runs)
        if args.ctl_only:
            continue

        # ---- scoring: pure CPU over the run files the subprocesses wrote ----
        if args.score_vs:
            reference = f"{run_base(name, args.data)}-{args.score_vs}"
            if not os.path.exists(run_path(args.out, reference)):
                print(f"[bench] no reference run file {reference!r} in {args.out}; skipping scoring")
                continue
            candidates = [n for _, n, _ in rows if n in done]
            ctl = ctl_run_name(name, args.data, spec.dtype_control)
            ctl = ctl if os.path.exists(run_path(args.out, ctl)) else None
        else:
            reference = rows[0][1] if rows[0][1] in done else None
            ctl = next((n for label, n, _ in rows if label == "ctl" and n in done), None)
            candidates = [n for label, n, _ in rows[1:] if label != "ctl" and n in done]
        if reference is None:
            continue
        if not candidates:                   # single-backend run: raw states, no verdicts
            score_runs(spec, args.out, reference, None)
            continue
        for cand in candidates:
            score_runs(spec, args.out, cand, reference, ctl=ctl)

    if failed_runs:
        for name, label, rc in failed_runs:
            print(f"[bench] FAILED run: spec={name} run={label} rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

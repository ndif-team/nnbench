"""Single benchmark entrypoint — replaces the 5 scripts/smoke_*.py.

Runs one spec (or all) through the one-pass sweep: amortized model load, warm-timed cells, oracle
under each workload regime, then the applicability map + performance table.

    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/bench.py --spec steering_gpt2
    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/bench.py --spec all --backends hf

Must be run under the `if __name__ == "__main__"` guard (vLLM EngineCore uses spawn).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.specs import SPECS  # noqa: E402
from isb.sweep import run_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="all", help="spec name or 'all'")
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    ap.add_argument("--serve", default=None,
                    help="nnsight-vllm-serve URL for the vllm_serve backend (e.g. http://server:6677)")
    ap.add_argument("--dump-refs", default=None, metavar="DIR",
                    help="persist this run's HF reference per (workload,label) to DIR for later VM scoring")
    ap.add_argument("--dump-ctl-refs", default=None, metavar="DIR",
                    help="GPU step: cache fp32-vLLM outputs to DIR so a later serve run can disambiguate "
                         "a bf16 precision near-tie (SUPPORTED_DEGRADED) from a real bug; skips the sweep")
    ap.add_argument("--refs", default=None, metavar="DIR",
                    help="score serve cells against cached integrated references loaded from DIR")
    ap.add_argument("--ctl-refs", default=None, metavar="DIR",
                    help="cached fp32-vLLM control outputs (from --dump-ctl-refs) for serve precision disambig")
    args = ap.parse_args()

    names = list(SPECS) if args.spec == "all" else [args.spec]
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown spec {name!r}; choices: {', '.join(SPECS)} or 'all'")
        print(f"\n########## spec: {name} ##########")
        if args.dump_ctl_refs:                       # standalone GPU step: cache fp32-vLLM, no sweep
            from isb.sweep.driver import dump_control_refs
            dump_control_refs(SPECS[name], args.dump_ctl_refs)
            continue
        run_sweep(SPECS[name], backends=tuple(args.backends),
                  serve_host=args.serve, dump_refs=args.dump_refs, refs=args.refs, ctl_refs=args.ctl_refs)


if __name__ == "__main__":
    main()

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
    args = ap.parse_args()

    names = list(SPECS) if args.spec == "all" else [args.spec]
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown spec {name!r}; choices: {', '.join(SPECS)} or 'all'")
        print(f"\n########## spec: {name} ##########")
        run_sweep(SPECS[name], backends=tuple(args.backends))


if __name__ == "__main__":
    main()

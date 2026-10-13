"""Score one executed run against another: pure CPU, no models, re-runnable any time.

    python scripts/score.py --case jacobian_lens/gpt2 --candidate vllm-cand --reference hf-ref
    # optional precision disambiguation from a control-dtype run (itself just an executed run):
    python scripts/score.py --case steering/gpt2 --candidate vllm-bf16 --reference hf-ref --ctl vllm-fp32

The axis is derived from provenance: cross-engine -> correctness; same engine -> config/topology
equivalence. The header names both stacks (nnsight commit, engine versions, host GPU).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402
from isb.runfile import INBOX  # noqa: E402
from isb.workloads import WORKLOADS, case  # noqa: E402
from isb.sweep.score import score_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, metavar="WORKLOAD/MODEL",
                    help="a workload (isb/workloads/) on a model (isb/models.py)")
    ap.add_argument("--data", default=None, metavar="SOURCE[:N]",
                    help="the same data binding the runs were executed with")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ctl", default=None,
                    help="run name of a control-dtype execution for precision disambiguation")
    ap.add_argument("--out", default=INBOX, help="directory holding the run files")
    args = ap.parse_args()

    spec = case(args.case)
    if spec is None:
        raise SystemExit(f"unknown case {args.case!r}; use <workload>/<model> with a workload of "
                         f"{', '.join(WORKLOADS)} and a model from isb/models.py")
    if args.data:
        from isb.data import DataRef
        from isb.sweep.spec import spec_with_data
        src, _, n = args.data.partition(":")
        spec = spec_with_data(spec, DataRef(src, int(n) if n else None))
    score_runs(spec, args.out, args.candidate, args.reference, ctl=args.ctl)


if __name__ == "__main__":
    main()

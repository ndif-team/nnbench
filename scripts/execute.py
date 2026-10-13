"""Execute ONE fully described run: (case × data × engine/deployment config) -> outputs + provenance.

Legacy standalone executor, directly callable inside the optional legacy backend containers.
The main scripts/bench.py runner instead invokes backends/NAME/run.py using frozen job inputs:

    python scripts/execute.py --case jacobian_lens/gpt2 \
        --engine transformers --name hf-ref
    python scripts/execute.py --case jacobian_lens/gpt2 \
        --engine vllm --name vllm-cand
    # then compare (no GPU needed):  python scripts/score.py --case jacobian_lens/gpt2 \
        --candidate vllm-cand --reference hf-ref

Every run writes ONE self-contained file, <out>/<name>.pt (cell outputs + provenance: client
stack, deployment, engine, host hardware). Fresh runs default into the inbox (runs/inbox);
archive one by moving the file into a collection directory. `--release` verifies the environment
is clean first and refuses on contamination; its findings are recorded into the provenance it
certifies. Must run under the `if __name__ == "__main__"` guard (vLLM EngineCore uses spawn).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.runfile import INBOX  # noqa: E402
from isb.runs import DeploymentConfig, EngineConfig, RunConfig  # noqa: E402
from isb.workloads import WORKLOADS, case  # noqa: E402
from isb.sweep.execute import execute_run  # noqa: E402


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, metavar="WORKLOAD/MODEL",
                    help="a workload (isb/workloads/) on a model (isb/models.py)")
    ap.add_argument("--data", default=None, metavar="SOURCE[:N]",
                    help="rebind the case's regimes to a registered data source")
    ap.add_argument("--realizations", default="default", choices=["default", "all"],
                    help="also run every alternative realization the cells declare")
    ap.add_argument("--engine", required=True, choices=["transformers", "vllm"])
    ap.add_argument("--engine-mode", default="async", choices=["async", "sync"])
    ap.add_argument("--deployment", default="local", choices=["local", "serve", "ndif"])
    ap.add_argument("--host", default=None, help="serve URL / ndif endpoint")
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="engine param (repeatable): dtype=float32, tensor_parallel_size=2, "
                         "enable_prefix_caching=true, ...")
    ap.add_argument("--name", required=True, help="run name; keys the output + provenance files")
    ap.add_argument("--out", default=INBOX, help="output dir; default is the inbox")
    ap.add_argument("--release", action="store_true",
                    help="verify the environment is clean first; refuse on contamination")
    ap.add_argument("--debug", action="store_true",
                    help="also write the instrumented debug companion (debug/<name>-debug.pt): "
                         "token ids, per-layer residuals, final logits over the first prompts; "
                         "diff two companions with scripts/debug_compare.py")
    args = ap.parse_args()

    spec = case(args.case)
    if spec is None:
        raise SystemExit(f"unknown case {args.case!r}; use <workload>/<model> with a workload of "
                         f"{', '.join(WORKLOADS)} and a model from isb/models.py")
    import dataclasses
    spec = dataclasses.replace(spec, realizations=args.realizations)
    if args.data:
        from isb.data import DataRef
        from isb.sweep.spec import spec_with_data
        src, _, n = args.data.partition(":")
        spec = spec_with_data(spec, DataRef(src, int(n) if n else None))

    findings = None
    if args.release:
        from isb.preflight import check_environment
        findings = check_environment(args.out)
        if findings:
            for x in findings:
                print(f"[release] environment not clean: {x}")
            raise SystemExit(2)

    run = RunConfig(
        engine=EngineConfig(args.engine, mode=args.engine_mode,
                            params={k: _coerce(v) for k, v in
                                    (kv.split("=", 1) for kv in args.param)}),
        deployment=DeploymentConfig(kind=args.deployment, host=args.host),
    )
    execute_run(spec, run, args.out, args.name, release_findings=findings, debug=args.debug)


if __name__ == "__main__":
    main()

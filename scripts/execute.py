"""Execute ONE fully described run: (spec × data × engine/deployment config) -> outputs + provenance.

    # the HF reference stack, in the transformers-testing env:
    CUDA_VISIBLE_DEVICES=4 /path/envs/nnsight-tf/bin/python scripts/execute.py \
        --spec jacobian_lens_gpt2 --engine transformers --name hf-ref
    # the vLLM candidate, in the vllm env:
    CUDA_VISIBLE_DEVICES=4 /path/envs/nnsight-vllm/bin/python scripts/execute.py \
        --spec jacobian_lens_gpt2 --engine vllm --name vllm-cand
    # then compare (no GPU needed):  python scripts/score.py --spec jacobian_lens_gpt2 \
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
from isb.specs import SPECS  # noqa: E402
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
    ap.add_argument("--spec", required=True)
    ap.add_argument("--data", default=None, metavar="SOURCE[:N]",
                    help="rebind the spec's regimes to a registered data source")
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

    if args.spec not in SPECS:
        raise SystemExit(f"unknown spec {args.spec!r}; choices: {', '.join(SPECS)}")
    spec = SPECS[args.spec]
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

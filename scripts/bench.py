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
from isb.specs import SPECS, default_specs  # noqa: E402
from isb.sweep import run_sweep  # noqa: E402


def _pp_backend_factory(pp, tp, executor, gpu_mem=0.2, max_model_len=None):
    """Backend factory for PP/TP equivalence mode. Both runs use the SAME dtype (spec.dtype_control)
    and memory knobs so the only variable is parallelism:
        "vllm_async" -> single-GPU (1,1)        the control / ground truth (GT2)
        "vllm_pp"    -> (tensor=tp, pipeline=pp) the candidate, scored against (1,1)
    'executor' must be "ray" for multi-node (it spreads PP/TP stages across the Ray cluster's nodes).
    gpu_mem/max_model_len matter for large models: the (1,1) control must hold the whole model on one
    GPU, so a multi-B model needs a high gpu_memory_utilization (and a capped max_model_len keeps KV
    small enough to fit alongside the weights).
    """
    from isb.backends import VLLMAsyncBackend

    def factory(name, sp):
        dt = sp.dtype_control
        common = dict(dtype=dt, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len)
        if name == "vllm_async":
            return VLLMAsyncBackend(**common)
        if name == "vllm_pp":
            return VLLMAsyncBackend(
                pipeline_parallel_size=pp,
                tensor_parallel_size=tp,
                distributed_executor_backend=executor,
                **common,
            )
        raise ValueError(f"unexpected backend {name!r} in PP/TP equivalence mode")

    return factory


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
    # ---- parallelism-equivalence mode (PP/TP) --------------------------------------------------
    # PP>1 or TP>1 switches the sweep to score a parallel vLLM engine against single-GPU vLLM
    # instead of against HF: control = vLLM (1,1) [GT2: same intervention at tp=1,pp=1], candidate =
    # vLLM (tp,pp). No HF reference is run. This is the right correctness model for pipeline/tensor
    # parallelism (where bitwise equivalence to HF is neither expected nor the question).
    ap.add_argument("--pp", type=int, default=1, help="pipeline_parallel_size (>1 enables PP/TP equivalence mode)")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (>1 enables PP/TP equivalence mode)")
    ap.add_argument("--executor", default=None, choices=["ray", "mp"],
                    help="vLLM distributed_executor_backend for the (tp,pp) candidate; 'ray' for multi-node")
    ap.add_argument("--gpu-mem", type=float, default=0.2,
                    help="gpu_memory_utilization for PP/TP mode (raise for large models; 0.2 suits gpt2)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="cap max_model_len in PP/TP mode (keeps KV small so big weights + KV fit one GPU)")
    args = ap.parse_args()

    parallel = args.pp > 1 or args.tp > 1

    names = default_specs() if args.spec == "all" else [args.spec]
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown spec {name!r}; choices: {', '.join(SPECS)} or 'all'")
        if args.dump_ctl_refs:                       # standalone GPU step: cache fp32-vLLM, no sweep
            print(f"\n########## spec: {name} ##########")
            from isb.sweep.driver import dump_control_refs
            dump_control_refs(SPECS[name], args.dump_ctl_refs)
            continue
        if parallel:
            print(f"\n########## spec: {name} "
                  f"[PP={args.pp} TP={args.tp} executor={args.executor or 'mp'}] ##########")
            run_sweep(SPECS[name], backends=("vllm_async", "vllm_pp"),
                      backend_factory=_pp_backend_factory(
                          args.pp, args.tp, args.executor, args.gpu_mem, args.max_model_len),
                      control="vllm_async")
            continue
        print(f"\n########## spec: {name} ##########")
        run_sweep(SPECS[name], backends=tuple(args.backends),
                  serve_host=args.serve, dump_refs=args.dump_refs, refs=args.refs, ctl_refs=args.ctl_refs)


if __name__ == "__main__":
    main()

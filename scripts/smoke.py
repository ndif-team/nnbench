"""Smoke-tier run (design.md §2 Micro/Method, §8.1): logit-lens on GPT-2,
HF vs vLLM-async, producing an applicability map per workload.

Runs every workload in configs/workload/ (or --workload <file> ...). Each backend is
loaded once and reused across workloads.

Run (env with nnsight=dev + vllm):
    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke.py
    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke.py --backends hf
"""
import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.report import print_map  # noqa: E402
from isb.resolve import GPT2, HF, VLLM_ASYNC  # noqa: E402
from isb.runner import evaluate, run_cell  # noqa: E402
from isb.spec.schema import load_yaml  # noqa: E402

BACKENDS = {
    "hf": (HF, "isb.backends.hf:HFBackend"),
    "vllm_async": (VLLM_ASYNC, "isb.backends.vllm_async:VLLMAsyncBackend"),
}


def _impl(dotted: str):
    mod, cls = dotted.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    ap.add_argument("--workload", nargs="*", default=None,
                    help="workload yaml files; default = all in configs/workload/")
    args = ap.parse_args()

    files = (
        [Path(p) for p in args.workload]
        if args.workload
        else sorted((ROOT / "configs/workload").glob("*.yaml"))
    )
    workloads = [load_yaml(str(f)) for f in files]
    family = GPT2  # smoke is GPT-2; the sweep uses family_for(config.model_type)

    # Isolate each (workload, backend) cell: a fresh engine per vLLM cell, so a
    # workload that crashes the EngineCore cannot poison the next one.
    for wl in workloads:
        cells = []
        for name in args.backends:
            profile, dotted = BACKENDS[name]
            print(f"\n>>> cell: {wl.id} on {name} ...")
            cells.append(run_cell(wl, args.repo, family, profile, _impl(dotted)))
        evaluate(cells, reference="hf")
        print_map(wl, args.repo, cells)


if __name__ == "__main__":
    main()

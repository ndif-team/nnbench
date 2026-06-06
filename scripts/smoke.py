"""Smoke-tier run (design.md §2 Micro/Method, §8.1): logit-lens on GPT-2,
HF vs vLLM-async, producing an applicability map.

Run (env with nnsight=dev + vllm):
    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test \
        python scripts/smoke.py

Pass `--backends hf` to run HF only.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.report import print_map  # noqa: E402
from isb.resolve import GPT2, HF, VLLM_ASYNC  # noqa: E402
from isb.runner import evaluate, run_cell  # noqa: E402
from isb.spec.schema import Workload, load_yaml  # noqa: E402

BACKENDS = {
    "hf": (HF, "isb.backends.hf:HFBackend"),
    "vllm_async": (VLLM_ASYNC, "isb.backends.vllm_async:VLLMAsyncBackend"),
}


def _impl(dotted: str):
    mod, cls = dotted.split(":")
    import importlib

    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai-community/gpt2")
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    ap.add_argument(
        "--workload", default=str(ROOT / "configs/workload/logit_lens.yaml")
    )
    args = ap.parse_args()

    wl = load_yaml(args.workload)
    family = GPT2  # smoke is GPT-2; family_for(config.model_type) used in the sweep

    cells = []
    for name in args.backends:
        profile, dotted = BACKENDS[name]
        print(f"\n>>> running cell: {name} ...")
        cells.append(run_cell(wl, args.repo, family, profile, _impl(dotted)))

    cells = evaluate(cells, reference="hf")
    print_map(wl, args.repo, cells)


if __name__ == "__main__":
    main()

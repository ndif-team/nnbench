"""Smoke-tier run (design.md §12): logit-lens on GPT-2, HF (control) vs vLLM-async,
across two variance params (`unembed=module` idiomatic, `unembed=weight` portable).

    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke.py
    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke.py --backends hf
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.backends import HFBackend, VLLMAsyncBackend  # noqa: E402
from isb.report import print_map  # noqa: E402
from isb.runner import evaluate, run_cell  # noqa: E402

METHOD = "logit_lens"
FAMILY = "gpt2"
REPO = "openai-community/gpt2"
PROMPTS = ["The Eiffel Tower is in the city of"]

# Variances: same methodology, same family, different observe/formulation params.
TASKS = [
    ({"unembed": "module"}, "unembed=module (idiomatic)"),
    ({"unembed": "weight"}, "unembed=weight (portable)"),
]
BACKEND_IMPLS = {"hf": HFBackend, "vllm_async": VLLMAsyncBackend}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    args = ap.parse_args()

    for params, label in TASKS:
        cells = []
        for name in args.backends:
            impl = BACKEND_IMPLS[name]()
            print(f"\n>>> cell: {METHOD}/{FAMILY}/{name} [{label}] ...")
            cells.append(
                run_cell(METHOD, FAMILY, name, impl, REPO, PROMPTS, params=params, label=label)
            )
        evaluate(cells, control="hf")
        print_map(METHOD, FAMILY, label, REPO, cells)


if __name__ == "__main__":
    main()

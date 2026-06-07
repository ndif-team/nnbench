"""Smoke-tier run: logit-lens on the LLAMA architecture, HF (control) vs vLLM-async.

This is the first SECOND-family cell — the real test of the per-family control (§12.2): the
vLLM-llama cell is scored against HF-llama, never against GPT-2. It exercises explicit-per-family
module naming (`model.model.layers` / `model.model.norm`, RMSNorm not LayerNorm) and asks whether
the backend-vs-HF delta holds for a different architecture than GPT-2.

Concrete weights = `HuggingFaceTB/SmolLM2-135M-Instruct`, a `LlamaForCausalLM` (`model_type:
llama`, tied embeddings). We use it rather than `meta-llama/Llama-3.x` because the meta-llama repos
are GATED and (on this machine) their tokenizers are not in the local HF cache, so they cannot load
offline. SmolLM2 drives the identical Llama code path in both nnsight backends, so the
`(methodology, family=llama, backend)` map cell is faithful; `print_map` shows the exact model id.

    CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 conda run -n nnsight-serve-test python scripts/smoke_llama.py
    CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 conda run -n nnsight-serve-test python scripts/smoke_llama.py --backends hf
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
FAMILY = "llama"
REPO = "HuggingFaceTB/SmolLM2-135M-Instruct"  # a LlamaForCausalLM; meta-llama is gated + tokenizer not cached
PROMPTS = ["The Eiffel Tower is in the city of"]

# Same params go to BOTH backends; per-backend cell DEFAULTS supply the right residual form
# (HF default residual="plain", vLLM default residual="fused"). We never force residual="fused" on
# HF — HF blocks can return (hidden, past_kv) and hidden+past_kv is garbage; "plain" is correct there.
TASKS = [
    ({"unembed": "module"}, "unembed=module (idiomatic)"),                       # vLLM: lm_head guarded -> ERROR
    ({"unembed": "weight"}, "unembed=weight (backend-aware portable)"),          # vLLM uses fused residual -> SUPPORTED
    ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain (naive GPT-2 port)"),  # vLLM -> SILENTLY_WRONG
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

"""Smoke-tier run for ACTIVATION PATCHING (causal tracing), GPT-2, HF vs vLLM-async.

Capture the residual at block L from a CLEAN run ("...France...") and transplant it into the same
block of a CORRUPTED run ("...Russia...") via two single-prompt traces (`be.patch`); observe the
corrupted run's next-token logits. HF is the per-family control; vLLM-async is the system under
test. The two-trace form means NO multi-invoke/barrier is needed, which is the whole reason this can
run on vLLM at all.

Non-vacuity guard: patching is only a meaningful test if it CHANGES the corrupted output (else a
backend that drops the patch would falsely score SUPPORTED). We compute the corrupted run's
UNPATCHED logits once (layer-independent) and report TV(unpatched, HF-patched) per layer.

    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/smoke_patching.py
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.backends import HFBackend, VLLMAsyncBackend  # noqa: E402
from isb.methodologies.logit_lens import _resid  # noqa: E402
from isb.oracle.equivalence import compare  # noqa: E402
from isb.report import print_map  # noqa: E402
from isb.runner import disambiguate_precision, evaluate, run_cell  # noqa: E402

METHOD = "activation_patching"
FAMILY = "gpt2"
REPO = "openai-community/gpt2"
CLEAN = "The capital of France is the city of"      # 8 tokens; minimal pair, differs only at
CORRUPTED = "The capital of Russia is the city of"  # the country token -> localized patch effect
PROMPTS = [CLEAN, CORRUPTED]

TASKS = [
    ({"layer": 3, "residual": "plain"}, "layer=3"),
    ({"layer": 9, "residual": "plain"}, "layer=9"),
]
BACKEND_IMPLS = {"hf": HFBackend, "vllm_async": VLLMAsyncBackend}


def _unpatched_corrupted_logits(residual="plain"):
    """Corrupted run's natural last-token logits (no patch). Layer-independent -> compute once."""
    impl = HFBackend()
    model = impl.load(REPO)
    try:
        def build():
            h, ln_f, head = model.transformer.h, model.transformer.ln_f, model.lm_head
            with torch.no_grad():
                normed = ln_f(_resid(h[-1].output, residual))
                return impl.last(F.linear(normed, head.weight))
        return impl.run(model, [CORRUPTED], build)
    finally:
        impl.teardown(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    args = ap.parse_args()

    unpatched = _unpatched_corrupted_logits() if "hf" in args.backends else None

    for params, label in TASKS:
        cells = []
        for name in args.backends:
            impl = BACKEND_IMPLS[name]()
            print(f"\n>>> cell: {METHOD}/{FAMILY}/{name} [{label}] ...")
            cells.append(
                run_cell(METHOD, FAMILY, name, impl, REPO, PROMPTS, params=params, label=label)
            )
        hf_cell = next((c for c in cells if c.backend == "hf"), None)
        hf_val = hf_cell.value if hf_cell is not None else None  # keep: evaluate() clears .value

        if unpatched is not None and hf_val is not None:
            eff = compare(unpatched, hf_val)
            strong = eff["top1_agree"] < 0.5 or eff["tv"] > 0.2
            verdict = "OK — patch moves the control" if strong else "WEAK — patch barely changes output"
            print(f"[non-vacuity guard | {label}] corrupted unpatched vs HF-patched: "
                  f"top1={eff['top1_agree']:.2f} tv={eff['tv']:.3f} -> {verdict}")

        evaluate(cells, control="hf")  # strict oracle: SUPPORTED or SILENTLY_WRONG
        # dtype control: a strict gate-failure may be bf16-vs-fp32 precision, not a bug (F-8)
        disambiguate_precision(
            cells, hf_val,
            lambda c: run_cell(METHOD, FAMILY, c.backend, VLLMAsyncBackend(dtype="float32"),
                               REPO, PROMPTS, params=params, label=label).value)
        print_map(METHOD, FAMILY, label, REPO, cells)


if __name__ == "__main__":
    main()

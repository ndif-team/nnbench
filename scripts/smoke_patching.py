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
from isb.oracle.equivalence import compare, is_equivalent  # noqa: E402
from isb.report import print_map  # noqa: E402
from isb.runner import evaluate, run_cell  # noqa: E402
from isb.states import AppState  # noqa: E402

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
        vllm_cell = next((c for c in cells if c.backend == "vllm_async"), None)
        hf_val = hf_cell.value if hf_cell is not None else None  # keep: evaluate() clears .value

        if unpatched is not None and hf_val is not None:
            eff = compare(unpatched, hf_val)
            strong = eff["top1_agree"] < 0.5 or eff["tv"] > 0.2
            verdict = "OK — patch moves the control" if strong else "WEAK — patch barely changes output"
            print(f"[non-vacuity guard | {label}] corrupted unpatched vs HF-patched: "
                  f"top1={eff['top1_agree']:.2f} tv={eff['tv']:.3f} -> {verdict}")

        evaluate(cells, control="hf")  # strict oracle: SUPPORTED or SILENTLY_WRONG

        # Dtype control: a strict gate-failure may be PRECISION (vLLM default bf16 vs HF fp32), not a
        # mechanism bug. Re-run vLLM at fp32; if it then matches HF, the bf16 divergence is precision
        # -> SUPPORTED_DEGRADED, not SILENTLY_WRONG. This is the disambiguation a fair cross-backend
        # oracle needs (F-8); without it, precision is mislabeled as a correctness bug.
        if vllm_cell is not None and vllm_cell.state == AppState.SILENTLY_WRONG and hf_val is not None:
            print(f"[dtype control | {label}] vLLM failed strict gate; re-checking at fp32 ...")
            fp32 = run_cell(METHOD, FAMILY, "vllm_async", VLLMAsyncBackend(dtype="float32"),
                            REPO, PROMPTS, params=params, label=label)
            if fp32.value is not None:
                m = compare(hf_val, fp32.value)
                if is_equivalent(m):
                    vllm_cell.state = AppState.SUPPORTED_DEGRADED
                    vllm_cell.metrics["fp32_tv"] = round(m["tv"], 4)
                    print(f"   vLLM-fp32 vs HF: top1={m['top1_agree']:.2f} tv={m['tv']:.4f} "
                          f"-> bf16 divergence is PRECISION -> SUPPORTED_DEGRADED")
                else:
                    print(f"   vLLM-fp32 vs HF: top1={m['top1_agree']:.2f} tv={m['tv']:.4f} "
                          f"-> persists in fp32 -> stays SILENTLY_WRONG")

        print_map(METHOD, FAMILY, label, REPO, cells)


if __name__ == "__main__":
    main()

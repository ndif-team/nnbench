"""Smoke-tier run for the ABLATION methodology, GPT-2, HF vs vLLM-async.

Zero a component (MLP or attention output) at block L and read the next-token distribution. HF is
the per-family control; vLLM-async is the system under test (whole-tuple replacement write).

Effect-size guard: an ablation that doesn't move the output makes the verdict vacuous, so we compute
the un-ablated baseline once (`target="none"`, layer-independent) and report TV(baseline, ablated).

    CUDA_VISIBLE_DEVICES=5 conda run -n nnsight-serve-test python scripts/smoke_ablation.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import isb.methodologies  # noqa: F401,E402  (registers the cells)
from isb.backends import HFBackend, VLLMAsyncBackend  # noqa: E402
from isb.oracle.equivalence import compare  # noqa: E402
from isb.report import print_map  # noqa: E402
from isb.runner import disambiguate_precision, evaluate, run_cell  # noqa: E402

METHOD = "ablation"
FAMILY = "gpt2"
REPO = "openai-community/gpt2"
PROMPTS = ["The Eiffel Tower is in the city of"]
LAYER = 6

TASKS = [
    ({"layer": LAYER, "target": "mlp"}, "target=mlp"),
    ({"layer": LAYER, "target": "attn"}, "target=attn"),
]
BACKEND_IMPLS = {"hf": HFBackend, "vllm_async": VLLMAsyncBackend}


def _baseline():
    """Un-ablated HF final logits (target='none'); layer-independent -> compute once."""
    return run_cell(METHOD, FAMILY, "hf", HFBackend(), REPO, PROMPTS,
                    params={"layer": LAYER, "target": "none"}, label="baseline").value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    args = ap.parse_args()

    base = _baseline() if "hf" in args.backends else None

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
        if base is not None and hf_val is not None:
            eff = compare(base, hf_val)
            strong = eff["top1_agree"] < 0.5 or eff["tv"] > 0.2
            verdict = "OK — ablation moves the control" if strong else "WEAK — ablation barely changes output"
            print(f"[effect-size guard | {label}] HF un-ablated vs ablated: "
                  f"top1={eff['top1_agree']:.2f} tv={eff['tv']:.3f} -> {verdict}")
        evaluate(cells, control="hf")
        # dtype control: a near-tie SILENTLY_WRONG may be bf16-vs-fp32 precision, not a bug (cf. F-8)
        disambiguate_precision(
            cells, hf_val,
            lambda c: run_cell(METHOD, FAMILY, c.backend, VLLMAsyncBackend(dtype="float32"),
                               REPO, PROMPTS, params=params, label=label).value)
        print_map(METHOD, FAMILY, label, REPO, cells)


if __name__ == "__main__":
    main()

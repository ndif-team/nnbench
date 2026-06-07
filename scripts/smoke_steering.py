"""Smoke-tier run for the STEERING methodology (design.md §12; first write methodology).

Steer GPT-2 block 8 toward the unembed direction of " Rome", read the final-layer portable-unembed
next-token distribution. HF is the per-family control; vLLM-async is the system under test. Two
variance params exercise the inplace-vs-replace write divergence.

A write methodology's verdict is only meaningful if the write has a detectable effect on the
CONTROL — otherwise a backend that silently no-ops the write would falsely score SUPPORTED. So we
first run an unsteered baseline (alpha=0) and report TV(baseline, HF-steered) as an effect-size
guard before trusting any SUPPORTED / SILENTLY_WRONG label.

    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke_steering.py
    CUDA_VISIBLE_DEVICES=0 conda run -n nnsight-serve-test python scripts/smoke_steering.py --backends hf
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
from isb.runner import evaluate, run_cell  # noqa: E402

METHOD = "steering"
FAMILY = "gpt2"
REPO = "openai-community/gpt2"
PROMPTS = ["The Eiffel Tower is in the city of"]

LAYER, TARGET, ALPHA = 8, " Rome", 6.0
TV_TOL = 0.05  # the oracle's equivalence tolerance (isb/oracle + runner default)

# Variances: same methodology/family, the write FORM differs (in-place vs whole-tuple replacement).
TASKS = [
    ({"layer": LAYER, "target": TARGET, "alpha": ALPHA, "mode": "inplace"}, "mode=inplace"),
    ({"layer": LAYER, "target": TARGET, "alpha": ALPHA, "mode": "replace"}, "mode=replace"),
]
BACKEND_IMPLS = {"hf": HFBackend, "vllm_async": VLLMAsyncBackend}


def _effect_size(mode):
    """TV(unsteered HF, steered HF) in the SAME write `mode` as the cell under test.

    The guard must exercise the very mode it gates: a backend whose `inplace` path silently no-ops
    would score SUPPORTED against a `replace`-validated guard. So we run the alpha=0 baseline and the
    alpha>0 steered run both in `mode`, and report whether HF's own output actually moved.
    """
    base = run_cell(METHOD, FAMILY, "hf", HFBackend(), REPO, PROMPTS,
                    params={"layer": LAYER, "target": TARGET, "alpha": 0.0, "mode": mode},
                    label=f"alpha=0 baseline [{mode}]")
    steer = run_cell(METHOD, FAMILY, "hf", HFBackend(), REPO, PROMPTS,
                     params={"layer": LAYER, "target": TARGET, "alpha": ALPHA, "mode": mode},
                     label=f"alpha>0 steered [{mode}]")
    if base.value is None or steer.value is None:
        return {"error": base.error or steer.error}
    return compare(base.value, steer.value)


def _print_guard(mode, eff):
    if "error" in eff:
        print(f"\n[effect-size guard | {mode}] HF baseline run failed: {eff['error']}")
        return
    # the effect must be comfortably larger than the oracle tolerance, or a SUPPORTED below is vacuous
    strong = eff["top1_agree"] < 0.5 or eff["tv"] > 4 * TV_TOL
    verdict = "OK — steering moves the control" if strong \
        else f"WEAK — effect tv={eff['tv']:.3f} not >> tv_tol={TV_TOL}; verdict may be vacuous (raise alpha)"
    print(f"\n[effect-size guard | {mode}] HF unsteered vs steered: "
          f"top1_agree={eff['top1_agree']:.2f} tv={eff['tv']:.3f} -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["hf", "vllm_async"])
    args = ap.parse_args()

    for params, label in TASKS:
        mode = params["mode"]
        if "hf" in args.backends:
            _print_guard(mode, _effect_size(mode))  # per-mode: validates the exact mode being gated
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

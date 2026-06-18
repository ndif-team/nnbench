"""steering spec — collapses smoke_steering.py.

Baseline = alpha=0 (no write -> pure forward + readout); effect-size = TV(alpha=0, alpha=6) on the
control (the per-mode non-vacuity guard, now declarative).
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import BATCHED, PROBE

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

steering_gpt2 = CellConfig(
    name="steering_gpt2",
    methodology="steering", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", PROBE), Workload("batched", BATCHED)],
    tasks=[
        ({**_S, "mode": "inplace"}, "mode=inplace"),
        ({**_S, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_S, "alpha": 6.0, "mode": "replace"},
    ),
    # in-place residual write raises on vLLM inference tensors (replacement works); whole-tuple `replace` is the
    # working form (matches HF exactly, tv=0.000). Batched HF is SUPPORTED here (verified — the steer
    # dominates, so it is robust to the GPT-2 position artifact, unlike logit_lens/ablation batched).
    expected={
        ("vllm_async", "interactive", "mode=inplace"): "ERROR",
        ("vllm_async", "batched", "mode=inplace"): "ERROR",     # batched gated (+ in-place)
        ("vllm_async", "batched", "mode=replace"): "ERROR",     # batched gated (awaiting upstream fix)
        # sync: in-place still hits InferenceMode protection (engine-wide). Batched runs each prompt
        # as its own vLLM request (multi-invoke, no left-padding), so replace is SUPPORTED — like HF
        # here, where the steer dominates the position artifact (in-place still ERRORs).
        ("vllm_sync", "interactive", "mode=inplace"): "ERROR",
        ("vllm_sync", "batched", "mode=inplace"): "ERROR",
        ("vllm_sync", "batched", "mode=replace"): "SUPPORTED",
    },
)

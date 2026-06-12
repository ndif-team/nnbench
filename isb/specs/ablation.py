"""ablation spec — collapses smoke_ablation.py.

Baseline = target='none' (no knockout -> pure forward + readout); effect-size = TV(none, attn) on
the control.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import BATCHED, PROBE

ablation_gpt2 = CellConfig(
    name="ablation_gpt2",
    methodology="ablation", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", PROBE), Workload("batched", BATCHED)],
    tasks=[
        ({"layer": 6, "target": "mlp"}, "target=mlp"),
        ({"layer": 6, "target": "attn"}, "target=attn"),
    ],
    baseline=BaselineSpec(params={"layer": 6, "target": "none"}),
    effect=EffectSpec(
        baseline_params={"layer": 6, "target": "none"},
        perturbed_params={"layer": 6, "target": "attn"},
    ),
    # whole-tuple replacement knockout is faithful on vLLM at fp32; bf16 default is a near-tie
    # precision divergence -> SUPPORTED_DEGRADED (F-9). GPT-2 batched HF hits the position artifact
    # (F-batched) -> SILENTLY_WRONG.
    expected={
        ("vllm_async", "interactive", "target=mlp"): "SUPPORTED_DEGRADED",
        ("vllm_async", "interactive", "target=attn"): "SUPPORTED_DEGRADED",
        ("hf", "batched", "target=mlp"): "SILENTLY_WRONG",
        ("hf", "batched", "target=attn"): "SILENTLY_WRONG",
        ("vllm_async", "batched", "target=mlp"): "ERROR",       # batched gated (awaiting upstream fix)
        ("vllm_async", "batched", "target=attn"): "ERROR",
    },
)

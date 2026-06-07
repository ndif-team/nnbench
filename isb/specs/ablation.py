"""ablation spec — collapses smoke_ablation.py.

Baseline = target='none' (no knockout -> pure forward + readout); effect-size = TV(none, attn) on
the control.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import BATCHED, ONE

ablation_gpt2 = CellConfig(
    name="ablation_gpt2",
    methodology="ablation", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", ONE), Workload("batched", BATCHED)],
    tasks=[
        ({"layer": 6, "target": "mlp"}, "target=mlp"),
        ({"layer": 6, "target": "attn"}, "target=attn"),
    ],
    baseline=BaselineSpec(params={"layer": 6, "target": "none"}),
    effect=EffectSpec(
        baseline_params={"layer": 6, "target": "none"},
        perturbed_params={"layer": 6, "target": "attn"},
    ),
)

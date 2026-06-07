"""steering spec — collapses smoke_steering.py.

Baseline = alpha=0 (no write -> pure forward + readout); effect-size = TV(alpha=0, alpha=6) on the
control (the per-mode non-vacuity guard, now declarative).
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import BATCHED, ONE

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

steering_gpt2 = CellConfig(
    name="steering_gpt2",
    methodology="steering", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", ONE), Workload("batched", BATCHED)],
    tasks=[
        ({**_S, "mode": "inplace"}, "mode=inplace"),
        ({**_S, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_S, "alpha": 6.0, "mode": "replace"},
    ),
)

"""activation-patching spec — collapses smoke_patching.py.

Interactive only: the workload prompts are the length-matched clean/corrupted pair the cell consumes
itself via two single-prompt traces (no batched workload in v1). Baseline = patch=False (corrupted
run, no transplant); effect-size = TV(unpatched, patched) on the control.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import CLEAN, CORRUPTED

activation_patching_gpt2 = CellConfig(
    name="activation_patching_gpt2",
    methodology="activation_patching", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", [CLEAN, CORRUPTED])],
    tasks=[
        ({"layer": 3, "residual": "plain"}, "layer=3"),
        ({"layer": 9, "residual": "plain"}, "layer=9"),
    ],
    baseline=BaselineSpec(params={"patch": False, "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"patch": False, "residual": "plain"},
        perturbed_params={"layer": 9, "residual": "plain", "patch": True},
    ),
)

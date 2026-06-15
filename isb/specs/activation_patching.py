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
    # the cell consumes its prompts as ONE clean/corrupt pair (not independent prompts), so no
    # per-prompt aggregation; multi-sample here would be N pairs (a follow-up).
    workloads=[Workload("interactive", [CLEAN, CORRUPTED], aggregate=False)],
    tasks=[
        ({"layer": 3, "residual": "plain"}, "layer=3"),
        ({"layer": 9, "residual": "plain"}, "layer=9"),
    ],
    baseline=BaselineSpec(params={"patch": False, "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"patch": False, "residual": "plain"},
        perturbed_params={"layer": 9, "residual": "plain", "patch": True},
    ),
    # The two-trace cross-prompt patch (whole-tuple replace) is the documented vLLM-correct recipe and
    # is faithful at fp32 (top1=1.00, tv≈0.001); at the bf16 default the patched top-1 flips on a
    # near-tie -> precision degradation, not a bug (the single-forward patch matches HF at fp32 but
    # flips a bf16 near-tie, separated by the dtype control).
    expected={
        ("vllm_async", "interactive", "layer=3"): "SUPPORTED_DEGRADED",
        ("vllm_async", "interactive", "layer=9"): "SUPPORTED_DEGRADED",
    },
)

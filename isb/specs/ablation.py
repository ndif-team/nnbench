"""ablation spec — collapses smoke_ablation.py.

Baseline = target='none' (no knockout -> pure forward + readout); effect-size = TV(none, attn) on
the control.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime

# data is a named, swappable source — see logit_lens.py. Default: clean prompts from the
# MIB circuit-track IOI snapshot (data/mib/README.md) — component knockout over the IOI
# task is the MIB ablation setting. The template bank stays available via --data factual.
PROBE = DataRef("mib/ioi_prompts", 100)
BATCHED = DataRef("mib/ioi_prompts", 16)

ablation_gpt2 = CellConfig(
    name="ablation_gpt2",
    methodology="ablation", family="gpt2", repo="openai-community/gpt2",
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
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

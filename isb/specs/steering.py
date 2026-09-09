"""steering spec — collapses smoke_steering.py.

Baseline = alpha=0 (no write -> pure forward + readout); effect-size = TV(alpha=0, alpha=6) on the
control (the per-mode non-vacuity guard, now declarative).
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime

# data is a named, swappable source — see logit_lens.py. Default: CounterFact factual-recall
# prompts (data/counterfact/README.md); the snapshot's per-item target_new strings are the
# hook for a later per-item steering-target upgrade (today `target` is one task param).
PROBE = DataRef("counterfact", 100)
BATCHED = DataRef("counterfact", 16)

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

steering_gpt2 = CellConfig(
    name="steering_gpt2",
    methodology="steering", family="gpt2", repo="openai-community/gpt2",
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
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

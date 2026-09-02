"""DAS spec (gpt2) — the gradient-training frontier with its portable apply form.

One workload, the whole labeled MIB IOI unit list in a single cell call (aggregate=False):
DAS trains ONE rotation over the train split and evaluates on the held-out units, so the
unit set cannot be split into independent per-unit traces. Two tasks split the methodology's
realizations: apply (seeded orthogonal rotation — COMPUTE ∘ replacement-WRITE, portable,
oracle-comparable across backends) and train (the gradient loop; HF-only, the `grad`
frontier's training-loop realization, guarded by held-out interchange accuracy).
Baseline = apply, so overhead-vs-baseline prices the training loop itself.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, ExecutionRegime

# labeled units: 16 train + 8 held-out (the cell's heldout default)
_UNITS = DataRef("mib/ioi_labeled", 24)

das_gpt2 = CellConfig(
    name="das_gpt2",
    methodology="das", family="gpt2", repo="openai-community/gpt2",
    regimes=[ExecutionRegime("interactive", _UNITS, aggregate=False)],
    tasks=[
        ({"train": 0}, "apply (seeded orthogonal rotation)"),
        ({"train": 24}, "train (24 rotation steps + held-out accuracy guard)"),
    ],
    baseline=BaselineSpec(params={"train": 0}),
    effect=None,
    # the train task runs 24 steps x 2 traces per timing trial; keep trials small
    warmup=1, n_trials=3,
)

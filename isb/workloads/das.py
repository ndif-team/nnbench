"""DAS: distributed alignment search, the gradient-training frontier with its portable apply form.

The whole labeled MIB IOI unit list goes to one cell call (aggregate=False): DAS trains ONE rotation
over the train split and evaluates on the held-out units, so the units cannot be split into
independent traces. Rows: apply (a seeded orthogonal rotation — COMPUTE ∘ replacement-WRITE, portable,
comparable across backends) and train (the gradient loop; HF-only, guarded by held-out interchange
accuracy). Baseline = apply, so overhead-vs-baseline prices the training loop itself.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Regime, Row, Workload

_HALF = Depth(1 / 2)

# 16 train + 8 held-out units (the cell's heldout default)
das = Workload(
    name="das", methodology="das",
    rows=(Row("apply a seeded orthogonal rotation at half depth", {"layer": _HALF, "train": 0}),
          Row("train the rotation at half depth (24 steps + held-out accuracy guard)",
              {"layer": _HALF, "train": 24})),
    regimes=(Regime("interactive", DataRef("mib/ioi_labeled", 24), aggregate=False),),
    baseline={"layer": _HALF, "train": 0},
    # the train row runs 24 steps x 2 traces per timing trial; keep trials small
    warmup=1, n_trials=3,
)

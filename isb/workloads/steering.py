"""steering: add a direction toward a target token's unembedding to the residual stream.

Strength is `alpha` times the stream norm, averaged over the forward's tokens (`mean_norm`, the
default) or per token (`token_norm`). alpha=6 saturates on the larger models (HF puts p=1.0000 on
' Rome' for every prompt), so those rows cannot tell a correct strength from a wrong one; the weaker
rows leave the output graded. Baseline = alpha=0 (no write -> pure forward + readout); effect-size =
TV(alpha=0, alpha=6) at half depth on the control. In-place vs replacement writes are a realization
the nnsight cells declare.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, Regime, Row, Workload

_ROME = {"target": " Rome"}
_SCALES = (("", {}), (", scaled by each token's norm", {"scale_by": "token_norm"}))
_QUARTER, _HALF = Depth(1 / 4), Depth(1 / 2)

# CounterFact factual-recall prompts (data/counterfact/README.md); the snapshot's per-item
# target_new strings are the hook for a per-item steering target later.
steering = Workload(
    name="steering", methodology="steering",
    rows=tuple(
        [Row(f"add toward ' Rome' at {where} depth{scale}",
             {**_ROME, "alpha": 6.0, "layer": depth, **knob})
         for where, depth in (("quarter", _QUARTER), ("half", _HALF)) for scale, knob in _SCALES]
        + [Row(f"add toward ' Rome' at half depth{scale}, alpha {alpha}",
               {**_ROME, "alpha": alpha, "layer": _HALF, **knob})
           for alpha in (0.1, 0.5) for scale, knob in _SCALES]),
    regimes=(Regime("interactive", DataRef("counterfact", 100)),
             Regime("batched", DataRef("counterfact", 16))),
    baseline={**_ROME, "alpha": 0.0, "layer": _HALF},
    effect=Effect(baseline={**_ROME, "alpha": 0.0, "layer": _HALF},
                  perturbed=(Row("alpha 6", {**_ROME, "alpha": 6.0, "layer": _HALF}),)),
)

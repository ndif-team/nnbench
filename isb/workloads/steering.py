"""steering: add a direction toward a target token's unembedding to the residual stream.

Strength is `alpha` times the stream norm, averaged over the forward's tokens (`mean_norm`, the
default) or per token (`token_norm`). alpha=6 saturates on the larger models (HF puts p=1.0000 on
' Rome' for every prompt), so those rows cannot tell a correct strength from a wrong one; the weaker
rows leave the output graded. Baseline = alpha=0 (no write -> pure forward + readout); effect-size =
TV(alpha=0, alpha=6) at half depth on the control. In-place vs replacement writes are a realization
the nnsight cells declare.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, FirstLayers, Regime, Row, Workload

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
    # evaluated over datasets, and applied to live traffic
    load_patterns=("offline", "concurrent", "open"),
    effect=Effect(baseline={**_ROME, "alpha": 0.0, "layer": _HALF},
                  perturbed=(Row("alpha 6", {**_ROME, "alpha": 6.0, "layer": _HALF}),)),
)


# How the cost of one forward grows with the number of edited layers: the "half depth" row's edit
# (alpha 6, mean norm) applied at every layer of the first k, k from one layer to every layer below
# the final (readout) layer. Baseline and effect check sit at the first layer, which every row
# edits, so a system that sizes its engine to the locations a case touches (nnsight's taps) sees
# exactly each row's layers.
_FIRST = FirstLayers(count=1)
steering_layers = Workload(
    name="steering_layers", methodology="steering",
    rows=tuple(
        Row(f"add toward ' Rome' at {where}", {**_ROME, "alpha": 6.0, "layers": layers})
        for where, layers in (
            ("the first layer", FirstLayers(count=1, below_final=True)),
            ("each of the first seventh of the layers", FirstLayers(1 / 7, below_final=True)),
            ("each of the first quarter of the layers", FirstLayers(1 / 4, below_final=True)),
            ("each of the first half of the layers", FirstLayers(1 / 2, below_final=True)),
            ("each of the first three quarters of the layers", FirstLayers(3 / 4, below_final=True)),
            ("every layer below the final layer", FirstLayers(1.0, below_final=True)))),
    regimes=(Regime("interactive", DataRef("counterfact", 100)),),
    baseline={**_ROME, "alpha": 0.0, "layers": _FIRST},
    effect=Effect(baseline={**_ROME, "alpha": 0.0, "layers": _FIRST},
                  perturbed=(Row("alpha 6 at the first layer", {**_ROME, "alpha": 6.0, "layers": _FIRST}),)),
)

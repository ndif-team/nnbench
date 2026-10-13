"""ablation: zero one block component and read the output.

Rows name the component. A model whose blocks split into attention and MLP sublayers runs the
`attn`/`mlp` rows; a single-op block family (Nemotron's `mixer`, design.md §12.7) runs the `mixer`
rows, where zeroing the mixer makes that layer the identity. A row a model cannot express is
recorded as not applicable. Baseline = target='none' (no knockout -> pure forward + readout);
effect-size = TV(none, knockout) on the control.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, Regime, Row, Workload

_HALF = Depth(1 / 2)
_SUBLAYERS = (("the attention output", "attn"), ("the MLP output", "mlp"))

# Clean prompts from the MIB circuit-track IOI snapshot (data/mib/README.md): component knockout
# over the IOI task is the MIB ablation setting.
ablation = Workload(
    name="ablation", methodology="ablation",
    rows=tuple(
        [Row(f"zero {what} at {where}", {"layer": layer, "target": target}, requires=(target,))
         for where, layer in (("layer 1", 1), ("half depth", _HALF)) for what, target in _SUBLAYERS]
        + [Row(f"zero the mixer at {where} depth", {"layer": Depth(f), "target": "mixer"},
               requires=("mixer",))
           for where, f in (("one third", 1 / 3), ("two thirds", 2 / 3))]),
    regimes=(Regime("interactive", DataRef("mib/ioi_prompts", 100)),
             Regime("batched", DataRef("mib/ioi_prompts", 16))),
    baseline={"layer": _HALF, "target": "none"},
    effect=Effect(baseline={"layer": _HALF, "target": "none"},
                  perturbed=(Row("attention", {"layer": _HALF, "target": "attn"}, requires=("attn",)),
                             Row("MLP", {"layer": _HALF, "target": "mlp"}, requires=("mlp",)),
                             Row("mixer", {"layer": Depth(1 / 3), "target": "mixer"},
                                 requires=("mixer",)))),
)

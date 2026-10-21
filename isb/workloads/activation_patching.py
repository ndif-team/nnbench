"""activation patching: transplant the clean run's residual into the corrupted run.

Interactive, multi-trace: the data is a SET of length-matched clean/corrupted pairs (the IOI task),
each a unit the cell consumes via two single-prompt traces. The verdict aggregates over all pairs
(top-1 fraction + mean TV), so a backend right on one pair but wrong on another is caught.
Baseline = patch=False (corrupted run, no transplant); effect-size = TV(unpatched, patched) at three
quarters depth on the control.

At bf16 the patched top-1 can flip on a near-tie: precision degradation, not a bug, separated by the
dtype control (the two-trace patch is faithful at fp32, top1=1.00, tv≈0.001).
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, Regime, Row, Workload

_QUARTER, _THREE_QUARTERS = Depth(1 / 4), Depth(3 / 4)

# The MIB circuit-track IOI snapshot (data/mib/README.md): clean prompt + s2_io_flip counterfactual,
# BPE-length-matched (also under the Qwen2.5 tokenizer, which the patch cell's shape check requires).
activation_patching = Workload(
    name="activation_patching", methodology="activation_patching",
    rows=tuple(
        Row(f"clean {what}residual into corrupt run at {where} depth", {"layer": depth, **knob})
        for what, knob in (("", {}), ("last-token ", {"positions": "last"}))
        for where, depth in (("quarter", _QUARTER), ("three quarters", _THREE_QUARTERS))),
    regimes=(Regime("interactive", DataRef("mib/ioi", 40)),),
    baseline={"patch": False},
    # causal tracing over a dataset of pairs
    load_patterns=("offline",),
    effect=Effect(baseline={"patch": False},
                  perturbed=(Row("patched", {"layer": _THREE_QUARTERS, "patch": True}),)),
)

"""gen_patching: the activation-patching transplant run under the decode loop.

Capture the clean residual, decode the corrupted prompt N steps with that residual injected at
prefill, score per-step logits. The data is a SET of length-matched IOI clean/corrupted pairs; each
pair is one trace unit and the verdict aggregates the per-step logits over all pairs. The
iteration-bound spelling is a realization the nnsight cells declare.

Baseline = patch=False (unpatched generation, same decode loop); effect-size = TV(unpatched,
patched) per step on the control. This is a COMPOSITION / step-lift law check (§3.7), not a new
primitive: it tests whether the transplant edge stays valid during decode. A SILENTLY_WRONG here is
a finding to investigate (real composition failure vs the cross-engine greedy-trajectory artifact),
not an assumed pass.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, Regime, Row, Workload

_THREE_QUARTERS = Depth(3 / 4)

# 12 MIB IOI pairs (data/mib/README.md) keep the generation regime (pair x 5 decode steps x trials)
# tractable while the verdict still aggregates over a set
gen_patching = Workload(
    name="gen_patching", methodology="gen_patching",
    rows=(Row("clean residual into corrupt decode at three quarters depth", {"layer": _THREE_QUARTERS}),),
    regimes=(Regime("generation", DataRef("mib/ioi", 12), new_tokens=5),),
    baseline={"layer": _THREE_QUARTERS, "patch": False},
    # causal tracing over a dataset, through the decode
    load_patterns=("offline",),
    effect=Effect(baseline={"layer": _THREE_QUARTERS, "patch": False},
                  perturbed=(Row("patched", {"layer": _THREE_QUARTERS, "patch": True}),)),
)

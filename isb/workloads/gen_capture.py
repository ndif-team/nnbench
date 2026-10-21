"""gen_capture: copy the residual stream off the device at every step of a greedy decode.

Export methodology, forward only: N greedy decode steps per prompt, the half-depth residual stream at
each step's newest position (the prefill's last prompt token, then each generated token) copied to
host memory inside the decode loop, as an online monitor of a generation does. Per-step rows are
stacked [new_tokens, d_model] and checked row per step. The iteration bound is a realization the
nnsight cells declare, as for gen_steering.

Read methodology -> no effect-size guard. Baseline = export=False: the same decode loop returning
only each step's greedy token, so overhead-vs-baseline is the per-step capture's cost inside the
generation regime.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Regime, Row, Workload

_HALF = Depth(1 / 2)

gen_capture = Workload(
    name="gen_capture", methodology="gen_capture",
    rows=(Row("half depth residual stream, every decode step", {"layer": _HALF, "export": True}),),
    # 32 CounterFact prompts, each greedily decoded for 8 steps (gen_steering's regime)
    regimes=(Regime("generation", DataRef("counterfact", 32), new_tokens=8),),
    baseline={"layer": _HALF, "export": False},
)

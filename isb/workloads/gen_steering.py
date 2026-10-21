"""gen_steering: the steering write applied at every step of a greedy decode.

N greedy decode steps per prompt, per-step logits stacked [new_tokens, vocab] and checked row per
step. The iteration-bound spelling (`iter[0:N]`, the vLLM working idiom, vs the documented `iter[:]`,
which drops every per-step save on vLLM) is a realization the nnsight cells declare.

Baseline = alpha=0 (no write, same decode loop) -> overhead isolates the steering write's cost inside
the generation regime; effect-size = TV(alpha=0, alpha=6) per step on the control.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Effect, Regime, Row, Workload

_ROME = {"target": " Rome"}
_HALF = Depth(1 / 2)

gen_steering = Workload(
    name="gen_steering", methodology="gen_steering",
    rows=tuple(
        Row(f"add toward ' Rome' at half depth, every step{scale}" + (f", alpha {alpha}" if alpha != 6.0 else ""),
            {**_ROME, "alpha": alpha, "layer": _HALF, **knob})
        for alpha in (6.0, 0.5)
        for scale, knob in (("", {}), (", scaled by each token's norm", {"scale_by": "token_norm"}))),
    # 32 CounterFact prompts, each greedily decoded with the per-step steer
    regimes=(Regime("generation", DataRef("counterfact", 32), new_tokens=8),),
    baseline={**_ROME, "alpha": 0.0, "layer": _HALF},
    # evaluated over datasets, and a guardrail on live traffic
    load_patterns=("offline", "concurrent", "open"),
    effect=Effect(baseline={**_ROME, "alpha": 0.0, "layer": _HALF},
                  perturbed=(Row("alpha 6", {**_ROME, "alpha": 6.0, "layer": _HALF}),)),
)

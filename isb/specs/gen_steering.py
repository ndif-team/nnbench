"""gen_steering specs, every model — the generation-time workload (catalog roadmap item 1).

First spec with a `generation` workload: N greedy decode steps per prompt, the steering write
applied at every step, per-step logits stacked [new_tokens, vocab] and oracle-checked row-per-step
(aggregate=True stacks the 8 probe prompts -> verdict over 8×new_tokens rows). The task axis is
the iteration-bound realization: `iter[0:N]` (the vLLM working idiom) vs `iter[:]` (the documented
idiom — the frontier marker where unbounded tracer.iter[:] drops all per-step saves on vLLM,
expected ERROR on vLLM until the upstream saves fix lands).

Baseline = alpha=0 (no write, same decode loop) -> overhead× isolates the steering write's cost
inside the generation regime; effect-size = TV(alpha=0, alpha=6) per step on the HF control.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime
from ._models import BF16, GPT2, QWEN25_7B, QWEN25_14B

# data is a named, swappable source: 32 CounterFact factual-recall prompts, each greedily decoded
# with the per-step steer; the verdict aggregates per-step logits over all of them
PROBE = DataRef("counterfact", 32)

# --- smoke (gpt2) -------------------------------------------------------------------------------

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

gen_steering_gpt2 = CellConfig(
    name="gen_steering_gpt2",
    methodology="gen_steering", family="gpt2", repo=GPT2,
    regimes=[ExecutionRegime("generation", PROBE, new_tokens=8)],
    tasks=[
        ({**_S, "bound": "bounded"}, "bound=iter[0:N]"),
        ({**_S, "bound": "unbounded"}, "bound=iter[:]"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "bound": "bounded"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "bound": "bounded"},
        perturbed_params={**_S, "bound": "bounded"},
    ),
    # unbounded iter[:] never sets a stop bound on the vLLM path -> the loop overruns and ALL
    # per-step saves are dropped (unbounded iter[:] drops all per-step saves on vLLM) -> clean ERROR. Bounded is the audit's prediction
    # (SUPPORTED via working idioms) — the composition this spec exists to measure.
)

# --- parallel (Qwen2.5-14B, GT2) ----------------------------------------------------------------

_L16 = {"layer": 16, "target": " Rome", "alpha": 6.0}

gen_steering_qwen = CellConfig(
    name="gen_steering_qwen",
    methodology="gen_steering", family="llama", repo=QWEN25_14B,
    regimes=[ExecutionRegime("generation", PROBE, new_tokens=8)],
    tasks=[
        ({**_L16, "bound": "bounded"}, "bound=iter[0:N]"),
        ({**_L16, "bound": "unbounded"}, "bound=iter[:]"),
    ],
    baseline=BaselineSpec(params={**_L16, "alpha": 0.0, "bound": "bounded"}),
    effect=EffectSpec(
        baseline_params={**_L16, "alpha": 0.0, "bound": "bounded"},
        perturbed_params={**_L16, "bound": "bounded"},
    ),
    dtype_control=BF16,
)

# --- comparison (Qwen2.5-7B, cross-system) ------------------------------------------------------

_STEER = {"target": " Rome", "alpha": 6.0}

cmp_gen_steering = CellConfig(
    name="cmp_gen_steering",
    methodology="gen_steering", family="llama", repo=QWEN25_7B,
    regimes=[ExecutionRegime("generation", DataRef("counterfact", 16), new_tokens=8)],
    tasks=[
        ({**_STEER, "layer": 14}, "add toward ' Rome' at layer 14, every step"),
        ({**_STEER, "layer": 14, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, every step, scaled by each token's norm"),
        ({**_STEER, "layer": 14, "alpha": 0.5}, "add toward ' Rome' at layer 14, every step, alpha 0.5"),
        ({**_STEER, "layer": 14, "alpha": 0.5, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, every step, scaled by each token's norm, alpha 0.5"),
    ],
    baseline=BaselineSpec(params={**_STEER, "layer": 14, "alpha": 0.0}),
    effect=EffectSpec(baseline_params={**_STEER, "layer": 14, "alpha": 0.0},
                      perturbed_params={**_STEER, "layer": 14}),
)

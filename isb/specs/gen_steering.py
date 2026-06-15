"""gen_steering spec — the generation-time workload (catalog roadmap item 1).

First spec with a `generation` workload: N greedy decode steps per prompt, the steering write
applied at every step, per-step logits stacked [new_tokens, vocab] and oracle-checked row-per-step
(aggregate=True stacks the 8 probe prompts -> verdict over 8×new_tokens rows). The task axis is
the iteration-bound realization: `iter[0:N]` (the vLLM working idiom) vs `iter[:]` (the documented
idiom — the frontier marker where unbounded tracer.iter[:] drops all per-step saves on vLLM,
expected ERROR on vLLM until the upstream saves fix lands).

Baseline = alpha=0 (no write, same decode loop) -> overhead× isolates the steering write's cost
inside the generation regime; effect-size = TV(alpha=0, alpha=6) per step on the HF control.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import PROBE

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

gen_steering_gpt2 = CellConfig(
    name="gen_steering_gpt2",
    methodology="gen_steering", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("generation", PROBE, new_tokens=8)],
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
    expected={
        ("vllm_async", "generation", "bound=iter[:]"): "ERROR",
    },
)

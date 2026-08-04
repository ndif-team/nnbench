"""gen_patching spec — the transplant edge run under the decode loop (catalog method-cell 4).

The cross-prompt transplant (activation patching) lifted into a greedy generation: capture the
clean residual, decode the corrupted prompt N steps with that residual injected at prefill, score
per-step logits. The workload is a SET of length-matched IOI clean/corrupted pairs; each pair is
one trace unit (the cell consumes it as its two prompts) and the verdict aggregates the per-step
logits over all pairs. Task axis = the iteration-bound realization:
`iter[0:N]` (the vLLM working idiom) vs `iter[:]` (the unbounded-iteration saves-drop frontier
marker, expected ERROR on vLLM).

Baseline = patch=False (unpatched generation, same decode loop) -> overhead× isolates the transplant's
cost in the generation regime; effect-size = TV(unpatched, patched) per step on the HF control.

Honest framing (see the methodology docstring): this is a COMPOSITION / step-lift law check, not a
new primitive — it tests whether the transplant edge stays valid run during decode (the §3.7 law),
and is the recipe a causalab `locate`-style analysis needs. The bf16 expectation mirrors the
single-forward patch (a near-tie precision degradation); the dtype control gives the mechanism
verdict at fp32. A SILENTLY_WRONG here is a finding to investigate (real composition
failure vs the cross-engine greedy-trajectory artifact), not an assumed pass.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ..data import DataRef

# data is a named, swappable pair source; 12 keeps the generation regime (pair x 5 decode steps
# x trials) tractable while the verdict still aggregates over a set instead of one hand-written pair
_PAIRS = DataRef("ioi_pairs", 12)

_P = {"layer": 9, "residual": "plain"}

gen_patching_gpt2 = CellConfig(
    name="gen_patching_gpt2",
    methodology="gen_patching", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("generation", _PAIRS, new_tokens=5, aggregate=True)],
    tasks=[
        ({**_P, "bound": "bounded"}, "bound=iter[0:N]"),
        ({**_P, "bound": "unbounded"}, "bound=iter[:]"),
    ],
    baseline=BaselineSpec(params={**_P, "patch": False, "bound": "bounded"}),
    effect=EffectSpec(
        baseline_params={**_P, "patch": False, "bound": "bounded"},
        perturbed_params={**_P, "patch": True, "bound": "bounded"},
    ),
)

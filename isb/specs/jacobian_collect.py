"""jacobian-collect spec (gpt2) — fitting the lens's transport maps, the many-VJP realization
of the `grad` frontier (the jacobian_lens spec only APPLIES fitted maps; this one produces them).

One workload, the whole wikitext prompt list in a single cell call (aggregate=False; J averages
over prompts). At gpt2 scale the sweep is 4 prompts x ceil(768/96)=8 batched backwards, gradients
read at all 11 source layers per backward. Baseline = grad=False (the same forward, no VJP — the
overhead denominator, and the part that also runs on vLLM). The produced [11, 768, 768] stack
exports to a fitted-lens artifact via scripts/export_jacobian.py ("file:" transport).
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, Workload

_PROMPTS = DataRef("wikitext", 4)

jacobian_collect_gpt2 = CellConfig(
    name="jacobian_collect_gpt2",
    methodology="jacobian_collect", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", _PROMPTS, aggregate=False)],
    tasks=[({"grad": True}, "collect (8 batched VJPs per prompt)")],
    baseline=BaselineSpec(params={"grad": False}),
    effect=None,
    # each timing trial runs the full 32-backward sweep; keep trials small
    warmup=1, n_trials=3,
)

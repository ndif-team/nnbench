"""Jacobian collection: fit the Jacobian lens's transport maps (the many-VJP realization of the
`grad` frontier; `jacobian_lens` only APPLIES fitted maps).

The whole wikitext prompt list goes to one cell call (aggregate=False; J averages over prompts). At
gpt2 scale the sweep is 4 prompts x ceil(768/96)=8 batched backwards, gradients read at all 11 source
layers per backward. Baseline = grad=False (the same forward, no VJP — the overhead denominator, and
the part that also runs on vLLM). The produced stack exports to a fitted-lens artifact via
scripts/export_jacobian.py ("file:" transport).
"""
from ..data import DataRef
from ..sweep.spec import Regime, Row, Workload

jacobian_collect = Workload(
    name="jacobian_collect", methodology="jacobian_collect",
    rows=(Row("collect the per-layer Jacobians (batched VJPs per prompt)", {"grad": True}),),
    regimes=(Regime("interactive", DataRef("wikitext", 4), aggregate=False),),
    baseline={"grad": False},
    # load_patterns: none (fitting is one call over the corpus; no load pattern)
    # each timing trial runs the full backward sweep; keep trials small
    warmup=1, n_trials=3,
)

"""Jacobian collection: fit the Jacobian lens's transport maps (the many-VJP realization of the
`grad` frontier; `jacobian_lens` only APPLIES fitted maps).

The whole wikitext prompt list goes to one cell call (aggregate=False; J averages over prompts). At
gpt2 scale the sweep is 4 prompts x ceil(768/96)=8 batched backwards, gradients read at all 11 source
layers per backward. Upstream (anthropics/jacobian-lens, jlens/fitting.py `fit`) runs prompts one at
a time and parallelizes by fitting disjoint prompt slices and merging the lenses, so one prompt per
call with the mean kept on the client is how the collection is run over a corpus: the offline load
at B=1. The batch axis is spent on cotangent rows, so B>1 per call is not the upstream shape. Baseline = grad=False (the same forward, no VJP — the overhead denominator, and
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
    # offline only, and at B=1 (docs/request-patterns.md): per-prompt maps are independent and
    # merged on the client; no deployment runs a collection per live request
    load_patterns=("offline",),
    # each timing trial runs the full backward sweep; keep trials small
    warmup=1, n_trials=3,
)

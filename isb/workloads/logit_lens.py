"""logit lens: decode every layer's residual stream through the final norm and unembed.

Read methodology -> no effect-size guard. The no-intervention baseline is a single-layer lens
(`layers=[-1]`), one read instead of the full stack, so overhead-vs-baseline reflects the per-layer
lens cost. The unembed spelling and the naive plain-residual port are realizations the nnsight cells
declare (isb/methodologies/logit_lens.py), not rows.
"""
from ..data import DataRef
from ..sweep.spec import Regime, Row, Workload

# Data is a named, swappable source (isb/data.py; override with --data). The default: CounterFact
# factual-recall prompts (data/counterfact/README.md); interactive scores over 100 units, batched
# pads a size-16 subset (batched is a regime check, not a volume axis).
logit_lens = Workload(
    name="logit_lens", methodology="logit_lens",
    rows=(Row("every layer, last token", {"layers": "all"}),),
    regimes=(Regime("interactive", DataRef("counterfact", 100)),
             Regime("batched", DataRef("counterfact", 16))),
    baseline={"layers": [-1]},
)

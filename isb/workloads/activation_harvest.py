"""activation harvest: copy the residual stream at every prompt token off the device.

Export methodology, forward only: one batched prefill whose half-depth residual stream is copied to
host memory at every prompt token, the collection step behind SAE training and probing at corpus
scale. The output is [tokens, d_model], each prompt's tokens in prompt order with padding dropped,
so HF's padded batch, vLLM's continuous batch and the per-prompt reference share one layout.
Read methodology -> no effect-size guard. Baseline = export=False: the same forward returning only
each prompt's greedy next token, so overhead-vs-baseline is the cost of reading the activations and
moving them off the device.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Regime, Row, Workload

_HALF = Depth(1 / 2)

# 16 WikiText paragraphs (data/wikitext/README.md), one padded batch
activation_harvest = Workload(
    name="activation_harvest", methodology="activation_harvest",
    rows=(Row("half depth residual stream, all valid tokens", {"layer": _HALF, "export": True}),),
    regimes=(Regime("batched", DataRef("wikitext", 16)),),
    baseline={"layer": _HALF, "export": False},
    # corpus-scale collection for SAE training and probing
    load_patterns=("offline",),
)

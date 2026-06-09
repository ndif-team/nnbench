"""attention-pattern spec (gpt2). Read methodology -> no effect-size guard.

Interactive-only: the output is `[n_layers, n_heads, k_len]` where `k_len` is the prompt length, so
a batched per-prompt reference (which would concat variable-length prompts on a shared axis) is
ill-defined here. The baseline reads a single layer (`layers=[0]`); the task reads all layers, so
overhead-vs-baseline reflects the per-layer attention-read cost.
"""
from ..sweep.spec import BaselineSpec, CellConfig, Workload
from ._prompts import ONE

attention_pattern_gpt2 = CellConfig(
    name="attention_pattern_gpt2",
    methodology="attention_pattern", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", ONE)],
    tasks=[({"layers": "all"}, "layers=all")],
    baseline=BaselineSpec(params={"layers": [0]}),
    effect=None,
)

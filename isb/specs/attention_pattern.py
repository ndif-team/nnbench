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
    # output is [layers, heads, k_len] — variable k_len across prompts can't be stacked, so no
    # per-prompt aggregation (the verdict already spans layers×heads).
    workloads=[Workload("interactive", ONE, aggregate=False)],
    tasks=[({"layers": "all"}, "layers=all")],
    baseline=BaselineSpec(params={"layers": [0]}),
    effect=None,
    # vLLM has no probability matrix to read (paged attention, no `.source` attention_interface op) —
    # a genuine architectural frontier, not a missing feature (attention weights have no denotation
    # under vLLM's paged attention). No working version exists.
    expected={
        ("vllm_async", "interactive", "layers=all"): "ERROR",
        ("vllm_sync", "interactive", "layers=all"): "ERROR",   # attn-weights frontier is engine-wide
    },
)

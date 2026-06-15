"""attribution-patching spec (gpt2). Gradient-based read -> no write to guard (effect=None).

Interactive only: the prompts are the length-matched clean/corrupt pair the cell consumes itself
(`be.attribute` runs two single-prompt traces). Baseline = `grad=False` (forward-only metric, no
backward — the overhead denominator, and the part that also runs on vLLM); the task does the full
forward+backward attribution. Output is a `[n_layers]` attribution vector.
"""
from ..sweep.spec import BaselineSpec, CellConfig, Workload
from ._prompts import CLEAN, CORRUPTED

attribution_patching_gpt2 = CellConfig(
    name="attribution_patching_gpt2",
    methodology="attribution_patching", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", [CLEAN, CORRUPTED], aggregate=False)],  # clean/corrupt pair
    tasks=[({"residual": "plain"}, "residual=plain")],
    baseline=BaselineSpec(params={"residual": "plain", "grad": False}),
    effect=None,
    # vLLM runs inference-mode (no autograd), so the backward an attribution needs raises — the whole
    # `grad` frontier is HF-only (gradients are unavailable on vLLM: inference mode, no autograd).
    # No working version exists on vLLM.
    expected={("vllm_async", "interactive", "residual=plain"): "ERROR"},
)

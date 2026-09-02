"""attribution-patching spec (gpt2). Gradient-based read -> no write to guard (effect=None).

Interactive, multi-trace: the workload is a set of labeled (clean, corrupted, answers) units
from the MIB IOI snapshot; each unit is its own two-trace attribution (`be.attribute`) whose
per-item answers feed the logit-difference metric, and the [n_layers] attribution vectors stack
across units for the verdict. Baseline = `grad=False` (forward-only metric, no backward — the
overhead denominator, and the part that also runs on vLLM); the task does the full
forward+backward attribution.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, ExecutionRegime

# labeled pairs: (clean, corrupted, (correct, incorrect)) — answers ride with the data
_PAIRS = DataRef("mib/ioi_labeled", 20)

attribution_patching_gpt2 = CellConfig(
    name="attribution_patching_gpt2",
    methodology="attribution_patching", family="gpt2", repo="openai-community/gpt2",
    regimes=[ExecutionRegime("interactive", _PAIRS, aggregate=True)],
    tasks=[({"residual": "plain"}, "residual=plain")],
    baseline=BaselineSpec(params={"residual": "plain", "grad": False}),
    effect=None,
    # vLLM runs inference-mode (no autograd), so the backward an attribution needs raises — the whole
    # `grad` frontier is HF-only (gradients are unavailable on vLLM: inference mode, no autograd).
    # No working version exists on vLLM.
)

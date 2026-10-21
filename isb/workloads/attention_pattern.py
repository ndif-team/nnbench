"""attention pattern: read every layer's attention probabilities. Read -> no effect-size guard.

Interactive-only: the output is `[n_layers, n_heads, k_len]` where `k_len` is the prompt length, so a
batched per-prompt reference (which would concat variable-length prompts on a shared axis) is
ill-defined, and per-prompt aggregation cannot stack. One prompt: an existence check. vLLM has no
probability matrix to read (paged attention, no `.source` attention_interface op), a genuine
architectural frontier, so the verdict is works-vs-ERROR and input volume adds nothing. The baseline
reads a single layer, so overhead-vs-baseline reflects the per-layer attention-read cost.
"""
from ..sweep.spec import Regime, Row, Workload

attention_pattern = Workload(
    name="attention_pattern", methodology="attention_pattern",
    rows=(Row("every layer", {"layers": "all"}),),
    regimes=(Regime("interactive", ["The Eiffel Tower is in the city of"], aggregate=False),),
    baseline={"layers": [0]},
    # load_patterns: none (docs/request-patterns.md): single stream only. Per-prompt outputs differ
    # in length, so a sweep's units cannot be stacked and scored; vLLM has no probability matrix to
    # read, so every pattern there would repeat the same ERROR.
)

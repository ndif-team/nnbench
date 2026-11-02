"""attention pattern: read every layer's attention probabilities. Read -> no effect-size guard.

Interactive-only: the output is `[n_layers, n_heads, k_len]` where `k_len` is the prompt length, so a
batched per-prompt reference (which would concat variable-length prompts on a shared axis) is
ill-defined, and per-prompt aggregation cannot stack. One prompt: an existence check. HF returns the
probabilities only under eager attention (the `nnsight-hf/eager` config is the reference); sdpa and
vLLM's paged attention never materialize them. On vLLM, gpt2 has no realization (the verdict is
works-vs-ERROR); Llama-tree models (Qwen2.5, Qwen3, Llama) compute the last query's softmax from the
attention kernel's query and key inputs. The baseline reads a single layer, so overhead-vs-baseline
reflects the per-layer attention-read cost.
"""
from ..sweep.spec import Regime, Row, Workload

attention_pattern = Workload(
    name="attention_pattern", methodology="attention_pattern",
    rows=(Row("every layer", {"layers": "all"}),),
    regimes=(Regime("interactive", ["The Eiffel Tower is in the city of"], aggregate=False),),
    baseline={"layers": [0]},
    # load_patterns: none (docs/request-patterns.md): single stream only. Per-prompt outputs differ
    # in length, so a sweep's units cannot be stacked and scored.
)

"""Shared prompt sets for the specs (kept tiny — this is a smoke/coverage harness, not a dataset)."""

ONE = ["The Eiffel Tower is in the city of"]

# Interactive probe set: independent factual completions with a clear top-1, run each as its OWN
# single-prompt trace (no batching) so a verdict aggregates top-1 agreement over N instead of judging
# one token. Used by the read/write last-token methodologies (logit-lens, steering, ablation).
PROBE = [
    "The Eiffel Tower is in the city of",
    "The capital of Japan is the city of",
    "Water is composed of hydrogen and",
    "The opposite of hot is",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "The first president of the United States was",
    "The author of Romeo and Juliet is",
]

# a small batched workload (throughput + per-prompt oracle under continuous batching)
BATCHED = [
    "The Eiffel Tower is in the city of",
    "The capital of Japan is the city of",
    "Water is composed of hydrogen and",
    "The opposite of hot is",
]

# minimal-pair clean/corrupted for activation patching (length-matched: differ only at the country)
CLEAN = "The capital of France is the city of"
CORRUPTED = "The capital of Russia is the city of"

"""Shared prompt sets for the specs (kept tiny — this is a smoke/coverage harness, not a dataset)."""

ONE = ["The Eiffel Tower is in the city of"]

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

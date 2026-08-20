"""Single-trace inputs for the attention-pattern frontier marker, plus one hand-written
minimal pair kept as a fixture for tests and the qwen generation spec.

- attention_pattern (ONE): an existence check — the attention-probability site is absent on vLLM
  (paged attention), so the verdict is works-vs-ERROR and input volume adds nothing.
- CLEAN/CORRUPTED: a length-matched minimal pair (differ only at the country). Attribution
  patching used to consume it with hardcoded answers; it now runs over labeled MIB IOI pairs
  (mib/ioi_labeled), whose per-item answers feed the metric.
"""

ONE = ["The Eiffel Tower is in the city of"]

# minimal-pair clean/corrupted for attribution patching (length-matched: differ only at the country)
CLEAN = "The capital of France is the city of"
CORRUPTED = "The capital of Russia is the city of"

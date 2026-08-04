"""Single-trace inputs for the two frontier-marker specs. Everything else uses the real trace
sets in `traces.py`; these two stay single-trace deliberately:

- attention_pattern (ONE): an existence check — the attention-probability site is absent on vLLM
  (paged attention), so the verdict is works-vs-ERROR and input volume adds nothing.
- attribution_patching (CLEAN/CORRUPTED): its metric hardcodes this pair's answers
  (" Paris"/" Moscow" logit difference), and its vLLM verdict is the `grad` frontier (ERROR,
  no autograd) regardless of inputs. A multi-pair form needs per-item answers — a real change to
  the methodology's metric, not a prompt swap.
"""

ONE = ["The Eiffel Tower is in the city of"]

# minimal-pair clean/corrupted for attribution patching (length-matched: differ only at the country)
CLEAN = "The capital of France is the city of"
CORRUPTED = "The capital of Russia is the city of"

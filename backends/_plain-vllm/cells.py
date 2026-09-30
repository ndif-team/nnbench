"""Plain vLLM expresses no intervention: every built-in methodology is declared unsupported, so the
backend contributes only its `__vanilla__` timing."""
from isb.methodologies.registry import Unsupported, cell
from isb.protocol import PROTOCOLS


def _unsupported(be, model, m, prompts, **params):
    raise Unsupported("plain vLLM has no interpretability layer; this backend only times the "
                      "no-intervention request")


for _methodology in PROTOCOLS:
    cell(_methodology, family="*", backend="vllm_plain")(_unsupported)

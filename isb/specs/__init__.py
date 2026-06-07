"""Spec registry — `bench.py --spec <name>` looks up here. Each spec is one CellConfig that
replaces a former `scripts/smoke_*.py`."""
from .ablation import ablation_gpt2
from .activation_patching import activation_patching_gpt2
from .logit_lens import logit_lens_gpt2, logit_lens_llama
from .steering import steering_gpt2

SPECS = {
    s.name: s
    for s in (
        logit_lens_gpt2,
        logit_lens_llama,
        steering_gpt2,
        activation_patching_gpt2,
        ablation_gpt2,
    )
}

__all__ = ["SPECS"]

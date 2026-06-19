"""Spec registry — `bench.py --spec <name>` looks up here. Each spec is one CellConfig that
replaces a former `scripts/smoke_*.py`."""
from .ablation import ablation_gpt2
from .activation_patching import activation_patching_gpt2
from .attention_pattern import attention_pattern_gpt2
from .attribution_patching import attribution_patching_gpt2
from .gen_patching import gen_patching_gpt2
from .gen_steering import gen_steering_gpt2
from .logit_lens import logit_lens_gpt2, logit_lens_llama
from .steering import steering_gpt2
from .qwen import (
    ablation_qwen,
    activation_patching_qwen,
    gen_steering_qwen,
    logit_lens_qwen,
    steering_qwen,
)
from .nemotron import (
    ablation_nemotron,
    ablation_nemotron_4b,
    logit_lens_nemotron,
    logit_lens_nemotron_4b,
    steering_nemotron,
    steering_nemotron_4b,
)

SPECS = {
    s.name: s
    for s in (
        logit_lens_gpt2,
        logit_lens_llama,
        steering_gpt2,
        gen_steering_gpt2,
        gen_patching_gpt2,
        activation_patching_gpt2,
        ablation_gpt2,
        attention_pattern_gpt2,
        attribution_patching_gpt2,
        # Qwen2.5-14B (family=llama) — large-model TP/PP equivalence specs
        logit_lens_qwen,
        steering_qwen,
        ablation_qwen,
        activation_patching_qwen,
        gen_steering_qwen,
        # Nemotron 3 Nano (family=nemotron) — hybrid Mamba; 4B dense (runnable) + 30B-A3B MoE (headline)
        logit_lens_nemotron_4b,
        steering_nemotron_4b,
        ablation_nemotron_4b,
        logit_lens_nemotron,
        steering_nemotron,
        ablation_nemotron,
    )
}

# The corpus swept by `bench.py --spec all`: the small smoke specs (gpt2 + SmolLM2-135M), which load
# on one modest GPU. Allowlist by design — large specs (Qwen 14B, and any later big model) are run by
# exact name and stay out of `all` automatically.
_DEFAULT_SPECS = (
    logit_lens_gpt2, logit_lens_llama, steering_gpt2, gen_steering_gpt2, gen_patching_gpt2,
    activation_patching_gpt2, ablation_gpt2, attention_pattern_gpt2, attribution_patching_gpt2,
)


def default_specs():
    """Spec names swept by `bench.py --spec all` — the small default corpus (large specs run by name)."""
    return [s.name for s in _DEFAULT_SPECS]


__all__ = ["SPECS", "default_specs"]

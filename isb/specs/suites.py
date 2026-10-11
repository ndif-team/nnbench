"""Named suites: which specs run together, and why. Specs live one file per methodology.

`bench.py --spec all` sweeps `smoke` only. Every other spec runs by exact name; an allowlist keeps
`all` at smoke scale, so a larger spec added later stays out of it automatically.
"""
from . import (ablation, activation_patching, attention_pattern, attribution_patching, das,
               gen_patching, gen_steering, jacobian_collect, jacobian_lens, logit_lens, steering)

# gpt2 + SmolLM2-135M: the default corpus, which loads on one modest GPU.
SMOKE = (
    logit_lens.logit_lens_gpt2, logit_lens.logit_lens_llama, jacobian_lens.jacobian_lens_gpt2,
    steering.steering_gpt2, gen_steering.gen_steering_gpt2, gen_patching.gen_patching_gpt2,
    activation_patching.activation_patching_gpt2, ablation.ablation_gpt2,
    attention_pattern.attention_pattern_gpt2, attribution_patching.attribution_patching_gpt2,
    das.das_gpt2, jacobian_collect.jacobian_collect_gpt2,
)

# Qwen2.5-14B-Instruct for the TP/PP parallelism-equivalence runs (`bench.py --pp 2 --tp 2`).
# family="llama" reuses the model.model.layers / model.model.norm / model.lm_head cells (Qwen2.5 is a
# LlamaForCausalLM-shaped decoder). dtype_control="bfloat16" so the (1,1) control fits on ONE 40GB
# A100 — fp32 14B (~56GB) would not. Interactive/generation regimes only (batched is gated on the
# vLLM async path regardless of model). These exist to exercise the GT2 oracle — candidate (tp,pp)
# vs single-GPU (1,1), same dtype — on a real multi-billion-param model. Under GT2 both sides run the
# SAME cell, so a cell only needs to RUN and be deterministic; absolute correctness vs HF (the
# fused-residual subtlety) is not what's scored here.
PARALLEL = (
    logit_lens.logit_lens_qwen, steering.steering_qwen, ablation.ablation_qwen,
    activation_patching.activation_patching_qwen, gen_steering.gen_steering_qwen,
)

# NVIDIA Nemotron 3 Nano, a hybrid Mamba-2 + sparse-attention decoder (design.md §12.7; loading in
# _models.py). The residual stream is additive across all block types, so the residual-stream
# methodologies port unchanged — logit_lens (read), steering (write), and ablation (zero a block's
# whole `.mixer` -> that layer becomes the identity; "which component" becomes "which LAYER").
# attention_pattern (only the few `*` layers have a matrix) and attribution's backward are the
# frontier and are not registered. Measured results for the 4B (2026-06-19) live in findings.md;
# every vLLM verdict there reduces to the documented gaps (interp-methods-catalog.md): guarded
# lm_head.forward, in-place-write restriction, fused-residual read. NemotronH adds none of its own.
# The 30B-A3B MoE specs need --pp/--tp to measure. dtype_control="bfloat16" (fp32 at this scale is
# impractical).
NEMOTRON_4B = (logit_lens.logit_lens_nemotron_4b, steering.steering_nemotron_4b,
               ablation.ablation_nemotron_4b)
NEMOTRON = (logit_lens.logit_lens_nemotron, steering.steering_nemotron, ablation.ablation_nemotron)

# Cross-system comparison (design.md §12.14) on Qwen2.5-7B-Instruct. Task parameters carry
# intervention semantics only (layer, direction, strength, component, positions). Steering strength
# is `alpha` times the stream norm, averaged over the forward's tokens (`mean_norm`, the default) or
# per token (`token_norm`); patching writes every position (`all`, the default) or the last one.
# Each system's cell picks that system's documented realization, so the same row is comparable
# across nnsight and foreign systems. Qwen2.5-7B-Instruct is a Llama-shaped decoder (family
# "llama"), 28 layers; the MIB IOI pairs are verified length-matched under the Qwen2.5 tokenizer.
# The rows were developed on Qwen2.5-1.5B-Instruct (same layer count and tokenizer; tied embeddings,
# which 7B does not have); the Delta run uses 7B because it is in the cluster's model cache.
COMPARISON = (
    logit_lens.cmp_logit_lens, steering.cmp_steering, gen_steering.cmp_gen_steering,
    activation_patching.cmp_activation_patching, ablation.cmp_ablation,
)

# Qwen3.5-4B with the upstream fitted Jacobian lens (4B-scale).
FITTED_JLENS = (jacobian_lens.jacobian_lens_qwen35,)

SUITES = {
    "smoke": SMOKE,
    "parallel": PARALLEL,
    "nemotron_4b": NEMOTRON_4B,
    "nemotron": NEMOTRON,
    "comparison": COMPARISON,
    "fitted_jlens": FITTED_JLENS,
}

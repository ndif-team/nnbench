"""Nemotron specs — NVIDIA **Nemotron 3 Nano (30B-A3B)**, the latest Nemotron family: a hybrid
Mamba-2 + sparse-attention **Mixture-of-Experts** decoder (design.md §12.7).

UNMEASURED FRONTIER. These specs declare the family and encode a hypothesis; no GPU sweep has run
yet, so every `expected` vLLM entry is a frontier *hypothesis* (held as ERROR so a future GPU run
surfaces whatever actually happens as a ⚠ surprise — the flip-detector convention in spec.py).

Architecture (config of nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B):
  - `architectures=["NemotronHForCausalLM"]`, `model_type="nemotron_h"`, ships repo-side custom code
    via `auto_map` -> needs `trust_remote_code=True` (wired through hf_kwargs/vllm_kwargs below).
  - 52 layers, hidden 2688, vocab 131072. `hybrid_override_pattern = "MEMEM*EMEMEM*EMEMEM*..."`:
    M=Mamba-2 SSM, E=MoE-FFN, *=sparse attention (only a handful of * layers).
  - MoE: `n_routed_experts=128`, `num_experts_per_tok=6`, `n_shared_experts=1`.
  - Module tree: `model.backbone.layers[i]` (each block ONE op, exposed as `.mixer`; additive
    residual `hidden = residual + mixer_out`; returns a single tensor) / `model.backbone.norm_f`
    (final RMSNorm) / `model.lm_head` (untied).

What ports and why (§12.7): logit_lens (residual READ), steering (residual WRITE), and ablation
(zero a block's whole `.mixer` -> that layer becomes identity) are RESIDUAL-STREAM ops, type-agnostic
across Mamba/attention/MLP/MoE — so they port to the HF control unchanged. attention_pattern (only
the few `*` layers have an attention matrix), attribution's backward (vLLM has no grad), and within-
block component targeting are the documented frontier and are not registered here. The CENTRAL open
question is whether nnsight-on-vLLM can trace NemotronH at all — its Mamba state lives in custom vLLM
kernels and nnsight's vLLM path was built around standard attention models.

Scale/precision: 30B MoE. The bf16 base repo is gated (HF auth); the FP8 variant
(`...-30B-A3B-FP8`) fits one large GPU but adds a quantization frontier. bf16 needs the parallelism
path (`bench.py --pp/--tp`, family runs under the GT2 oracle). dtype_control="bfloat16" (fp32 30B is
impractical).
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import PROBE

# bf16 base is gated (anonymous fetch 401s -> exists, needs HF auth); swap to "...-30B-A3B-FP8" for a
# single-GPU run (quantization frontier) — bf16 30B needs multi-GPU via --pp/--tp.
_REPO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
_TRC = {"trust_remote_code": True}        # NemotronH ships auto_map custom code
_BF16 = "bfloat16"
_S = {"layer": 16, "target": " Rome", "alpha": 6.0}

# vLLM entries are the UNMEASURED frontier hypothesis (see module docstring): held ERROR so a GPU run
# flags the real outcome. The open risk is NemotronH traceability under nnsight-vLLM; the idiomatic
# unembed and the in-place write are engine-wide ERRORs independent of that.

logit_lens_nemotron = CellConfig(
    name="logit_lens_nemotron",
    methodology="logit_lens", family="nemotron", repo=_REPO,
    workloads=[Workload("interactive", PROBE)],
    tasks=[
        ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain"),
        ({"unembed": "weight", "residual": "fused"}, "unembed=weight, residual=fused"),
        ({"unembed": "module", "residual": "plain"}, "unembed=module"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1], "residual": "plain"}),
    effect=None,
    dtype_control=_BF16,
    hf_kwargs=_TRC, vllm_kwargs=_TRC,
    expected={
        ("vllm_async", "interactive", "unembed=weight, residual=plain"): "ERROR",
        ("vllm_async", "interactive", "unembed=weight, residual=fused"): "ERROR",
        ("vllm_async", "interactive", "unembed=module"): "ERROR",   # guarded lm_head.forward + frontier
    },
)

steering_nemotron = CellConfig(
    name="steering_nemotron",
    methodology="steering", family="nemotron", repo=_REPO,
    workloads=[Workload("interactive", PROBE)],
    tasks=[
        ({**_S, "mode": "inplace"}, "mode=inplace"),
        ({**_S, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_S, "alpha": 6.0, "mode": "replace"},
    ),
    dtype_control=_BF16,
    hf_kwargs=_TRC, vllm_kwargs=_TRC,
    expected={
        ("vllm_async", "interactive", "mode=inplace"): "ERROR",   # in-place inference-tensor write
        ("vllm_async", "interactive", "mode=replace"): "ERROR",   # frontier: NemotronH traceability
    },
)

ablation_nemotron = CellConfig(
    name="ablation_nemotron",
    methodology="ablation", family="nemotron", repo=_REPO,
    workloads=[Workload("interactive", PROBE)],
    # target="mixer" zeroes the whole block's single op -> that layer becomes identity. Pick `layer`
    # to choose WHICH op type to knock out (the pattern says which indices are Mamba/attention/MoE).
    tasks=[
        ({"layer": 16, "target": "mixer"}, "layer=16 mixer"),
        ({"layer": 32, "target": "mixer"}, "layer=32 mixer"),
    ],
    baseline=BaselineSpec(params={"layer": 16, "target": "none"}),
    effect=EffectSpec(
        baseline_params={"layer": 16, "target": "none"},
        perturbed_params={"layer": 16, "target": "mixer"},
    ),
    dtype_control=_BF16,
    hf_kwargs=_TRC, vllm_kwargs=_TRC,
    expected={
        ("vllm_async", "interactive", "layer=16 mixer"): "ERROR",   # frontier: NemotronH traceability
        ("vllm_async", "interactive", "layer=32 mixer"): "ERROR",
    },
)

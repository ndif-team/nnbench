"""Nemotron specs — NVIDIA **Nemotron 3 Nano**, the latest Nemotron family: a hybrid Mamba-2 +
sparse-attention decoder (the 30B is also Mixture-of-Experts). See design.md §12.7.

Backend loading differs (measured 2026-06-19):
  - HF: transformers' BUILT-IN NemotronH, trust_remote_code=False. The repo's remote modeling
    hard-requires mamba-ssm; the built-in falls back to a naive torch path when mamba-ssm/causal-conv1d
    are absent ("fast path is not available ... naive implementation" warning) — no extra deps.
  - vLLM: its NATIVE NemotronH for compute, but its config validation refuses an auto_map repo unless
    trust_remote_code=True (vllm_kwargs below). That flag only permits reading the config; the remote
    modeling is never executed, so vLLM also needs no mamba-ssm.

HF and vLLM share the module tree `model.model.layers[i]` (each block ONE op exposed as `.mixer` with
a `.block_type`; additive residual) / `model.model.norm_f` / `model.lm_head` (untied); HF returns the
plain residual tensor, vLLM uses fused-residual RMSNorm (read with residual="fused").

Two registered sizes (both `NemotronHForCausalLM`, `model_type="nemotron_h"`):

  - `*_nemotron_4b`  -> NVIDIA-Nemotron-3-Nano-4B-BF16: 42 layers, hidden 3136, DENSE hybrid
    (`hybrid_override_pattern = "M-M-M-MM-M-M*-..."`; M=Mamba-2, -=MLP, *=attention). The cheap,
    GPU-runnable member used to MEASURE the family (no MoE to complicate the read/write).
  - `*_nemotron`     -> NVIDIA-Nemotron-3-Nano-30B-A3B-BF16: 52 layers, hidden 2688, MoE
    (`n_routed_experts=128`, top-6, +1 shared; pattern `"MEMEM*EMEM..."`, E=MoE-FFN). The headline
    "latest" target; needs the parallelism path (`bench.py --pp/--tp`) under the GT2 oracle.

What ports and why (§12.7): the residual stream is additive across all block types, so the
residual-stream methodologies port unchanged — logit_lens (read), steering (write), and ablation
(zero a block's whole `.mixer` -> that layer becomes the identity; "which component" becomes "which
LAYER"). attention_pattern (only the few `*` layers have a matrix) and attribution's backward are the
frontier and are not registered here.

Measured results for the 4B (2026-06-19) live in findings.md; every vLLM verdict there reduces to
the documented gaps (interp-methods-catalog.md): guarded lm_head.forward, in-place-write
restriction, fused-residual read. NemotronH adds none of its own. The `*_nemotron` (30B-A3B MoE)
specs need --pp/--tp to measure. dtype_control="bfloat16" (fp32 at this scale is impractical).
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ..data import DataRef

# data is a named, swappable source, sized for the 4B/30B nemotron runs
PROBE = DataRef("factual", 32)

_REPO_30B = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
_REPO_4B = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
_BF16 = "bfloat16"
# vLLM-only: its config validation refuses an auto_map repo without trust_remote_code, but it uses its
# NATIVE NemotronH for compute (no remote modeling executed, no mamba-ssm). HF stays on the built-in
# path (hf_kwargs left empty -> trust_remote_code=False).
_VLLM_TRC = {"trust_remote_code": True}
_S = {"layer": 16, "target": " Rome", "alpha": 6.0}


def _nemotron_specs(suffix: str, repo: str):
    """Build the (logit_lens, steering, ablation) CellConfigs for one Nemotron size — pure spec data,
    not a cell-construction layer (cells stay explicit per family, §12.1)."""
    logit_lens = CellConfig(
        name=f"logit_lens_nemotron{suffix}",
        methodology="logit_lens", family="nemotron", repo=repo,
        workloads=[Workload("interactive", PROBE)],
        tasks=[
            ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain"),
            ({"unembed": "weight", "residual": "fused"}, "unembed=weight, residual=fused"),
            ({"unembed": "module", "residual": "plain"}, "unembed=module"),
        ],
        baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1], "residual": "plain"}),
        effect=None,
        dtype_control=_BF16,
        vllm_kwargs=_VLLM_TRC,
    )
    steering = CellConfig(
        name=f"steering_nemotron{suffix}",
        methodology="steering", family="nemotron", repo=repo,
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
        vllm_kwargs=_VLLM_TRC,
    )
    ablation = CellConfig(
        name=f"ablation_nemotron{suffix}",
        methodology="ablation", family="nemotron", repo=repo,
        workloads=[Workload("interactive", PROBE)],
        # target="mixer" zeroes the block's single op -> that layer becomes identity. Pick `layer` to
        # choose WHICH op type to knock out (the pattern says which indices are Mamba/attention/MoE).
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
        vllm_kwargs=_VLLM_TRC,
    )
    return logit_lens, steering, ablation


logit_lens_nemotron, steering_nemotron, ablation_nemotron = _nemotron_specs("", _REPO_30B)
logit_lens_nemotron_4b, steering_nemotron_4b, ablation_nemotron_4b = _nemotron_specs("_4b", _REPO_4B)

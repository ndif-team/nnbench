"""Models the specs run on, and the load flags a model needs. Specs live one file per methodology;
which specs run together is in suites.py.

Nemotron 3 Nano (design.md §12.7) loads differently per backend (measured 2026-06-19):
  - HF: transformers' BUILT-IN NemotronH, trust_remote_code=False. The repo's remote modeling
    hard-requires mamba-ssm; the built-in falls back to a naive torch path when mamba-ssm/causal-conv1d
    are absent ("fast path is not available ... naive implementation" warning) — no extra deps.
  - vLLM: its NATIVE NemotronH for compute, but its config validation refuses an auto_map repo unless
    trust_remote_code=True (NEMOTRON_VLLM below). That flag only permits reading the config; the
    remote modeling is never executed, so vLLM also needs no mamba-ssm.

HF and vLLM share the module tree `model.model.layers[i]` (each block ONE op exposed as `.mixer` with
a `.block_type`; additive residual) / `model.model.norm_f` / `model.lm_head` (untied); HF returns the
plain residual tensor, vLLM uses fused-residual RMSNorm (read with residual="fused"). Two sizes (both
`NemotronHForCausalLM`, `model_type="nemotron_h"`):

  - NEMOTRON_4B: 42 layers, hidden 3136, DENSE hybrid (`hybrid_override_pattern = "M-M-M-MM-M-M*-..."`;
    M=Mamba-2, -=MLP, *=attention). The cheap, GPU-runnable member used to MEASURE the family.
  - NEMOTRON_30B: 52 layers, hidden 2688, MoE (`n_routed_experts=128`, top-6, +1 shared; pattern
    `"MEMEM*EMEM..."`, E=MoE-FFN). The headline target; needs `bench.py --pp/--tp` under GT2.
"""

GPT2 = "openai-community/gpt2"
SMOLLM2 = "HuggingFaceTB/SmolLM2-135M-Instruct"   # a LlamaForCausalLM; meta-llama is gated + uncached
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"            # family "llama": a Llama-shaped decoder, 28 layers
QWEN25_14B = "Qwen/Qwen2.5-14B-Instruct"          # family "llama" (profiles.py documents the reuse)
QWEN35_4B = "Qwen/Qwen3.5-4B"                     # family "qwen3_5": hybrid linear attention
NEMOTRON_30B = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
NEMOTRON_4B = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

BF16 = "bfloat16"
# vLLM-only: its config validation refuses an auto_map repo without trust_remote_code, but it uses its
# NATIVE NemotronH for compute (no remote modeling executed, no mamba-ssm). HF stays on the built-in
# path (hf_kwargs left empty -> trust_remote_code=False).
NEMOTRON_VLLM = {"trust_remote_code": True}

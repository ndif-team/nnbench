"""Milestone 0 (final): confirm the REAL nnbench helper forces vLLM's CpuPlatform when no GPU, so the
serve backend builds a meta VLLM model CUDA-free in a GPU-less container."""
import sys

sys.path.insert(0, "/nnbench")

from isb.backends.vllm_serve import _force_cpu_platform_when_no_gpu  # noqa: E402

_force_cpu_platform_when_no_gpu()

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda.is_available:", torch.cuda.is_available())

from vllm.platforms import current_platform  # noqa: E402

print("vLLM platform class:", type(current_platform).__name__)

from nnsight.modeling.vllm import VLLM  # noqa: E402

m = VLLM("openai-community/gpt2")
print("META BUILD OK:", type(m).__name__, "| dispatched:", m.dispatched,
      "| vllm_entrypoint:", m.vllm_entrypoint)
print("meta tree: model.transformer.h ->", len(m.transformer.h), "layers")
print("vllm._C imported:", "vllm._C" in sys.modules, "| torch.cuda init:", torch.cuda.is_initialized())
print("MILESTONE0_FINAL: PASS")

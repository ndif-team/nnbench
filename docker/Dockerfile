# Shared image for the VM-style nnbench split (server AND client).
#
# Both roles run the same software stack — the difference is purely runtime (see docker-compose.yml):
#   - server: gets a GPU, runs `nnsight.modeling.vllm.serve.cli` (CUDA platform, real engine)
#   - client: NO GPU (NVIDIA_VISIBLE_DEVICES=void) -> NVML sees 0 devices -> vLLM CPU platform ->
#             builds a meta VLLM model CUDA-free and submits traces over HTTP.
#
# The vLLM serve module + the meta-build path live in the local nnsight `dev` checkout (not on PyPI),
# so nnsight is mounted at runtime and put first on PYTHONPATH; `pip install nnsight` here only pulls
# nnsight's pure-Python runtime deps (astor, cloudpickle, httpx, python-socketio, pydantic, toml,
# rich, zstandard, ...) on top of the vLLM image's torch/transformers — it does not disturb vLLM.
#
# Slimming the CLIENT to a truly CUDA-free image (drop libcuda / use a +cpu vllm build) is a follow-up,
# gated on Milestone 0 step 3 (whether the CPU-platform meta build imports the libcuda-linked vLLM
# extensions). For now the client reuses this image and is made GPU-less at runtime.
FROM vllm/vllm-openai:v0.15.1

RUN pip install --no-cache-dir nnsight

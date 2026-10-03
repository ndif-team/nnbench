# Same engine build as nnsight-vllm and vllm-lens, so the systems differ only in the interpretability layer.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${VLLM_IMAGE}
# IBM/vLLM-Hook has no release on PyPI; pin the commit. Its requirement.txt allows vllm>=0.5,<=0.21.
ARG VLLM_HOOK_REF=0e34fdd
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone https://github.com/IBM/vLLM-Hook /opt/vllm-hook \
    && git -C /opt/vllm-hook checkout ${VLLM_HOOK_REF} \
    && python3 -m pip install --no-cache-dir --no-deps -e /opt/vllm-hook/vllm_hook_plugins \
    && python3 -m pip install --no-cache-dir "zstandard>=0.23.0" blake3 safetensors \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

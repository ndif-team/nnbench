# Same engine build as nnsight-vllm, so the two systems differ only in the interpretability layer.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${VLLM_IMAGE}
ARG VLLM_LENS_VERSION=1.2.1
# --no-deps keeps the image's vLLM; vllm-lens 1.2.1 declares vllm>=0.16.0.
RUN python3 -m pip install --no-cache-dir --no-deps "vllm-lens==${VLLM_LENS_VERSION}" \
    && python3 -m pip install --no-cache-dir "cloudpickle>=3.0.0" "zstandard>=0.23.0" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

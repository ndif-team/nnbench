# TransformerLens 4.0.0's [vllm] extra pins vllm>=0.20.2,<0.21 (validated on 0.20.2); the base
# image supplies that vLLM.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.20.2
FROM ${VLLM_IMAGE}
ARG TRANSFORMER_LENS_VERSION=4.0.0
RUN python3 -m pip install --no-cache-dir "transformer-lens==${TRANSFORMER_LENS_VERSION}" \
    && python3 -c "import vllm, transformer_lens; assert vllm.__version__ == '0.20.2', vllm.__version__" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

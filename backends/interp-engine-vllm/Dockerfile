# interp-engine pins vLLM exactly (its [vllm] extra is vllm==0.28.0); the base image supplies it.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.28.0
FROM ${VLLM_IMAGE}
ARG INTERP_ENGINE_VERSION=1.12.0
RUN python3 -m pip install --no-cache-dir "interp-engine==${INTERP_ENGINE_VERSION}" \
    && python3 -c "import vllm, interp_engine; assert vllm.__version__ == '0.28.0', vllm.__version__" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

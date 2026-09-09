ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${VLLM_IMAGE}
ARG NNSIGHT_REF=5166eb12ceb3b78091f7a8673f0cb47fd453dcb9
RUN apt-get update && apt-get install -y --no-install-recommends git gcc python3-dev
RUN python3 -m pip install --no-cache-dir \
      "nnsight @ git+https://github.com/ndif-team/nnsight.git@${NNSIGHT_REF}" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

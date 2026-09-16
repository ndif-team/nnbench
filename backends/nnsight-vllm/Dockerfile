ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${VLLM_IMAGE}
ARG NNSIGHT_REF=260c555bf2e3bd3395eed4df0435f35e253b2d3d
RUN apt-get update && apt-get install -y --no-install-recommends git gcc python3-dev
RUN python3 -m pip install --no-cache-dir \
      "nnsight @ git+https://github.com/ndif-team/nnsight.git@${NNSIGHT_REF}" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

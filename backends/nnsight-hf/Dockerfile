ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}
ARG NNSIGHT_REF=260c555bf2e3bd3395eed4df0435f35e253b2d3d
ARG TRANSFORMERS_VERSION=5.12.1
RUN apt-get update && apt-get install -y --no-install-recommends git gcc python3-dev
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      "torch==2.11.0" \
      "click==8.3.1" \
      "accelerate==1.15.0" \
      "nnsight @ git+https://github.com/ndif-team/nnsight.git@${NNSIGHT_REF}" \
      "transformers==${TRANSFORMERS_VERSION}" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs
RUN python3 -m pip check

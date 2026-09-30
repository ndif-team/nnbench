# Plain vLLM, nothing installed on top: the whole-system denominator (design.md §12.14). Each
# vllm-plain-* backend builds this file with its own engine version.
ARG VLLM_IMAGE
FROM ${VLLM_IMAGE}
RUN mkdir -p /models /outputs && chmod 0777 /models /outputs

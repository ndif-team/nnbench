# Same engine build as nnsight-vllm and vllm-lens, so the systems differ only in the interpretability layer.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.19.1
FROM ${VLLM_IMAGE}
# IBM/vLLM-Hook has no release on PyPI; pin the commit. At 0e34fdd (v0.2.0) the package is
# vllm_hook_plugins/ and its requirement.txt allows vllm>=0.5,<=0.21. From v1.0.0 the repository is
# MIA, installed from its root (setup.py: vllm-hook-mia, vllm==0.32.0), so the 1-0-1 image names the
# v1.0.1 commit and the vLLM 0.32.0 base. --no-deps keeps the base image's vLLM, torch and
# transformers, the ones its plain-vLLM floor runs.
ARG VLLM_HOOK_REF=0e34fdd
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone https://github.com/IBM/vLLM-Hook /opt/vllm-hook \
    && git -C /opt/vllm-hook checkout ${VLLM_HOOK_REF} \
    && if [ -f /opt/vllm-hook/setup.py ]; then package=/opt/vllm-hook; \
       else package=/opt/vllm-hook/vllm_hook_plugins; fi \
    && python3 -m pip install --no-cache-dir --no-deps -e "$package" \
    && python3 -m pip install --no-cache-dir "zstandard>=0.23.0" blake3 safetensors \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

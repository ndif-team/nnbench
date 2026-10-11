# TransformerLens 4.0.0's [vllm] extra pins vllm>=0.20.2,<0.21 (validated on 0.20.2); the base
# image supplies that vLLM.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.20.2
FROM ${VLLM_IMAGE}
ARG TRANSFORMER_LENS_VERSION=4.0.0
# TransformerLens needs transformers>=5.9.0 and the base ships 5.8.0, so pip upgrades it; pin the
# upgrade. transformers 5.18 dropped `PretrainedConfig._commit_hash`, which the worker reads as the
# model revision for the reference-compatibility check (isb/sweep/execute.py).
ARG TRANSFORMERS_VERSION=5.15.1
RUN python3 -m pip install --no-cache-dir "transformer-lens==${TRANSFORMER_LENS_VERSION}" \
      "transformers==${TRANSFORMERS_VERSION}" \
    && python3 -c "import vllm, transformer_lens; assert vllm.__version__ == '0.20.2', vllm.__version__" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs

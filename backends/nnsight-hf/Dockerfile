ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}
ARG NNSIGHT_REF=260c555bf2e3bd3395eed4df0435f35e253b2d3d
ARG TORCH_VERSION=2.11.0
ARG TRANSFORMERS_VERSION=5.12.1
ARG ACCELERATE_VERSION=1.15.0
ARG CLICK_VERSION=8.3.1
# base-image packages to remove before the check, when a pin above conflicts with them
ARG REMOVE_PACKAGES=none
RUN apt-get update && apt-get install -y --no-install-recommends git gcc python3-dev
# HF model loading uses device_map, which requires Accelerate independently of nnsight core.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      "click==${CLICK_VERSION}" \
      "accelerate==${ACCELERATE_VERSION}" \
      "nnsight @ git+https://github.com/ndif-team/nnsight.git@${NNSIGHT_REF}" \
      "transformers==${TRANSFORMERS_VERSION}" \
    && mkdir -p /models /outputs \
    && chmod 0777 /models /outputs
# nnsight 0.4.x opens nnsight.log inside its package directory at import; images run read-only, so
# point that file at /tmp. Releases without the file are untouched.
RUN python3 -c "import importlib.util as u, os; p = os.path.join(os.path.dirname(u.find_spec('nnsight').origin), 'nnsight.log'); os.path.exists(p) and (os.remove(p), os.symlink('/tmp/nnsight.log', p))"
RUN if [ "${REMOVE_PACKAGES}" != none ]; then python3 -m pip uninstall --break-system-packages -y ${REMOVE_PACKAGES}; fi
RUN python3 -m pip check

"""vLLM-Lens in its fastest documented realization: the same plugin and engine as the system's
backend (../backend.py), driving this config's cells.py, registered for `vllm_lens_opt`."""
from ..backend import VLLMLensBackend, create_backend as _create_backend


class VLLMLensOptBackend(VLLMLensBackend):
    name = "vllm_lens_opt"


def create_backend(spec):
    return _create_backend(spec, cls=VLLMLensOptBackend, mode="vllm-lens-opt")

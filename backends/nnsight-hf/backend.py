"""nnsight on HF transformers. The installed nnsight release selects the adapter: 0.4 builds an
intervention graph (nnsight04.py, with its own cells in cells_nnsight04.py), 0.5 iterates
generation with a context manager (nnsight05.py), later releases run the shared HFBackend."""
from importlib.metadata import version

from isb.backends.hf import HFBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.hf_kwargs, options())
    release = version("nnsight")
    if release.startswith("0.4."):
        from . import cells_nnsight04  # noqa: F401  (registers 0.4's cells)
        from .nnsight04 import NNsight04Backend as cls

        if params.pop("force_text_causal", False):
            raise ValueError("the nnsight 0.4 backend does not load multimodal-wrapper checkpoints")
    elif release.startswith("0.5."):
        from .nnsight05 import NNsight05HFBackend as cls
    else:
        cls = HFBackend
    return cls(**params), RunConfig(EngineConfig("transformers", params=params))

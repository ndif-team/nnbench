"""The nnsight-HF backend owns library selection and model-load translation."""
from isb.backends.hf import HFBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.hf_kwargs, options())
    return HFBackend(**params), RunConfig(EngineConfig("transformers", params=params))

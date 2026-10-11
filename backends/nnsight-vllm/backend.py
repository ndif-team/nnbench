"""The nnsight-vLLM backend owns library selection and model-load translation."""
from isb.backends.vllm_async import VLLMAsyncBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    # nnsight builds AsyncLLM with enforce_eager=not taps (modeling/vllm/vllm.py, _load_async);
    # taps are off here, so the engine runs eager.
    return VLLMAsyncBackend(**params), RunConfig(EngineConfig("vllm", params={**params,
                                                                              "enforce_eager": True}))

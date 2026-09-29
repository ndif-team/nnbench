"""The nnsight-vLLM backend owns library selection and model-load translation."""
from isb.backends.vllm_async import VLLMAsyncBackend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    return VLLMAsyncBackend(**params), RunConfig(EngineConfig("vllm", params=params))


if __name__ == "__main__":
    main(create_backend)

"""nnsight on vLLM through its synchronous engine, `VLLM(mode="sync")`, nnsight's documented default.

The other vLLM-plugin systems in the comparison (vLLM-Lens, TransformerLens, interp-engine's sync
facade) drive vLLM's synchronous `LLM`; `nnsight-vllm` drives nnsight's AsyncLLM path. This backend
measures nnsight through the same synchronous client path. Cells resolve through the vllm_async
fallback (design.md §12.14).
"""
from isb.backends.vllm_sync import VLLMSyncBackend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    # nnsight builds vllm.LLM with enforce_eager=not taps (modeling/vllm/vllm.py, _load_sync);
    # taps are off here, so the engine runs eager.
    return VLLMSyncBackend(**params), RunConfig(EngineConfig("vllm", mode="sync", params={
        **params, "enforce_eager": True}))


if __name__ == "__main__":
    main(create_backend)

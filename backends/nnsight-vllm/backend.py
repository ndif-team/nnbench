"""nnsight on vLLM. `mode` in the engine options selects the realization: `sync` (the default)
runs nnsight's `VLLM(mode="sync")`, the units of a call as one trace of several invokes; `async`
runs its AsyncLLM with one request per unit; `opt` runs installed edits and plain requests
(backend_opt.py, cells_opt.py). An installed nnsight 0.5 takes its adapter (nnsight05.py)."""
from importlib.metadata import version

from isb.backends.vllm_async import VLLMAsyncBackend
from isb.backends.vllm_sync import VLLMSyncBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    mode = params.pop("mode", "sync")
    if mode == "opt":
        from . import backend_opt

        return backend_opt.create(spec, params)
    if mode == "async":
        cls = VLLMAsyncBackend
    elif mode == "sync":
        cls = VLLMSyncBackend
        if version("nnsight").startswith("0.5."):
            from .nnsight05 import NNsight05SyncBackend as cls
    else:
        raise ValueError(f"mode must be sync, async or opt, not {mode!r}")
    # nnsight builds the engine with enforce_eager=not taps (nnsight modeling/vllm/vllm.py); no taps here.
    run = RunConfig(EngineConfig("vllm", mode=mode, params={**params, "enforce_eager": True}))
    return cls(**params), run

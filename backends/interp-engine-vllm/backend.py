"""The interp-engine backend: Neuronpedia / Decode Research's engine on vLLM.

The shared worker drives the cells in cells.py through this backend object. interp-engine's
methods are async and bound to one event loop; `sync_model` is its documented synchronous facade,
holding one loop for the model's life.
"""
import contextlib
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend, prompts_of
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class EngineModel:
    engine: object             # interp_engine SyncModel over the vLLM backend
    tokenizer: object          # the checkpoint's HF tokenizer, used exactly as the nnsight cells do
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    maps: dict = field(default_factory=dict)      # J-lens transport maps, built once per model


class InterpEngineBackend(Backend):
    name = "interp_engine"

    # The sync facade submits every call to one event loop on its own thread
    # (`interp_engine/_loop.py`, LoopRunner.run: run_coroutine_threadsafe per call, from any
    # thread), and `capture` and the per-request lens ops are documented as safe to call
    # concurrently with other requests (`vllm_backend.py`), so client threads calling the cells at
    # once overlap on the engine's scheduler (design.md §12.17); nothing to open.
    concurrent_requests = True

    @contextlib.contextmanager
    def concurrent_session(self):
        yield

    def __init__(self, backend="vllm", **engine_options):
        # backend: interp-engine's engine choice (load.py). "vllm" hooks the eager forward and
        # serves every point; "vllm-static" replays CUDA graphs over the taps it bakes in
        # (resid_post at every layer by default), its documented fast mode.
        self.backend = backend
        self.engine_options = engine_options

    def load(self, repo: str):
        from interp_engine import load_model, sync_model
        from transformers import AutoConfig, AutoTokenizer

        engine = sync_model(load_model(repo, backend=self.backend, **self.engine_options))
        engine.warmup()
        config = AutoConfig.from_pretrained(repo)
        return EngineModel(engine, AutoTokenizer.from_pretrained(repo), config,
                           config.num_hidden_layers)

    def plain(self, model, units, *, new_tokens=1):
        """`generate_text` with no capture points and no steering spec, one request per prompt,
        issued concurrently on the facade's loop for the scheduler to batch."""
        import asyncio

        ids = [model.tokenizer(prompt)["input_ids"] for prompt in prompts_of(units)]

        async def all_of():
            return await asyncio.gather(*(
                model.engine.model.generate_text(row, max_tokens=new_tokens, temperature=0.0)
                for row in ids))

        model.engine.runner.run(all_of())

    def collective_rpc(self, model, method, args=()):
        """`AsyncLLM.collective_rpc` of the engine interp-engine builds, one result per rank: the
        facade's `model` property gives the VLLMModel, whose `engine` attribute holds the AsyncLLM
        once `warmup` (in `load`) has built it (vllm_backend.py, `_ensure_engine`); the call runs on
        the facade's loop, the one that engine is bound to. `engine` is backend surface outside
        the sync protocol, as `unembed_rows` is; the same attribute serves `default` and
        `vllm-static`."""
        return model.engine.runner.run(model.engine.model.engine.collective_rpc(method, args=args))

    def teardown(self, model) -> None:
        model.engine.shutdown()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    backend = InterpEngineBackend(**params)
    run = RunConfig(EngineConfig("vllm", mode="interp-engine", params={
        **params, "enforce_eager": backend.backend == "vllm",
        # load_model's `num_gpus` is vLLM's tensor_parallel_size (load.py), default 1
        "tensor_parallel_size": int(params.get("num_gpus", 1)),
        "interp_engine": metadata.version("interp-engine")}))
    return backend, run

"""The interp-engine backend: Neuronpedia / Decode Research's engine on vLLM.

The shared worker drives the cells in cells.py through this backend object. interp-engine's
methods are async and bound to one event loop; `sync_model` is its documented synchronous facade,
holding one loop for the model's life.
"""
from dataclasses import dataclass
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


class InterpEngineBackend(Backend):
    name = "interp_engine"

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

    def teardown(self, model) -> None:
        model.engine.shutdown()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    backend = InterpEngineBackend(**params)
    run = RunConfig(EngineConfig("vllm", mode="interp-engine", params={
        **params, "enforce_eager": backend.backend == "vllm",
        "interp_engine": metadata.version("interp-engine")}))
    return backend, run

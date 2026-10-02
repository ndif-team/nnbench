"""The interp-engine backend: Neuronpedia / Decode Research's engine on vLLM (design.md §12.14).

The shared worker drives the cells in cells.py through this backend object. interp-engine's
methods are async and bound to one event loop; `sync_model` is its documented synchronous facade,
holding one loop for the model's life.
"""
from dataclasses import dataclass
from importlib import metadata

from isb.backends.base import Backend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class EngineModel:
    engine: object             # interp_engine SyncModel over the vLLM backend
    tokenizer: object          # the checkpoint's HF tokenizer, used exactly as the nnsight cells do
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    concurrent: bool = False   # issue independent captures concurrently (cells.py, patching)


class InterpEngineBackend(Backend):
    name = "interp_engine"

    def __init__(self, backend="vllm", concurrent_captures=False, **engine_options):
        # backend: interp-engine's engine choice (load.py). "vllm" hooks the eager forward and
        # serves every point; "vllm-static" replays CUDA graphs over the taps it bakes in
        # (resid_post at every layer by default), its documented fast mode.
        self.backend = backend
        self.concurrent_captures = concurrent_captures
        self.engine_options = engine_options

    def load(self, repo: str):
        from interp_engine import load_model, sync_model
        from transformers import AutoConfig, AutoTokenizer

        engine = sync_model(load_model(repo, backend=self.backend, **self.engine_options))
        engine.warmup()
        config = AutoConfig.from_pretrained(repo)
        return EngineModel(engine, AutoTokenizer.from_pretrained(repo), config,
                           config.num_hidden_layers, self.concurrent_captures)

    def vanilla(self, model, prompt, *, new_tokens):
        """generate_text with no capture points and no steering spec."""
        model.engine.generate_text(model.tokenizer(prompt)["input_ids"], max_tokens=new_tokens,
                                   temperature=0.0)

    def teardown(self, model) -> None:
        model.engine.shutdown()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    import cells  # noqa: F401  (registers the interp_engine cells)

    run = RunConfig(EngineConfig("vllm", mode="interp-engine", params={
        **params, "enforce_eager": params.get("backend", "vllm") == "vllm",
        "interp_engine": metadata.version("interp-engine")}))
    return InterpEngineBackend(**params), run


if __name__ == "__main__":
    main(create_backend)

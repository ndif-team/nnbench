"""The TransformerLens 4.0 vLLM bridge as a system under comparison (design.md §12.14).

`RemoteBridge.boot_vllm` keeps torch.compile and CUDA graphs: its capture and intervention hooks are
installed before compilation and apply `output * scale + bias`. The shared worker drives the cells
in cells.py through this backend object.
"""
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class BridgeModel:
    bridge: object             # transformer_lens RemoteBridge over vLLM
    tokenizer: object          # the checkpoint's HF tokenizer, used exactly as the nnsight cells do
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    weights: dict = field(default_factory=dict)   # unembedding and final-norm weight, fetched once


class TransformerLensBackend(Backend):
    name = "transformer_lens"

    def __init__(self, dtype: str, **engine_options):
        self.dtype = dtype
        self.engine_options = engine_options

    def load(self, repo: str):
        import torch
        from transformer_lens.model_bridge import RemoteBridge
        from transformers import AutoConfig, AutoTokenizer

        bridge = RemoteBridge.boot_vllm(repo, dtype=getattr(torch, self.dtype), **self.engine_options)
        config = AutoConfig.from_pretrained(repo)
        return BridgeModel(bridge, AutoTokenizer.from_pretrained(repo), config,
                           config.num_hidden_layers)

    def teardown(self, model) -> None:
        model.bridge.close()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    import cells  # noqa: F401  (registers the transformer_lens cells)

    run = RunConfig(EngineConfig("vllm", mode="transformer-lens", params={
        **params, "enforce_eager": False,
        "transformer_lens": metadata.version("transformer-lens")}))
    return TransformerLensBackend(**params), run


if __name__ == "__main__":
    main(create_backend)

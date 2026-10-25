"""The TransformerLens 4.0 vLLM bridge as a system under comparison.

`RemoteBridge.boot_vllm` keeps torch.compile and CUDA graphs: its capture and intervention hooks are
installed before compilation and apply `output * scale + bias`. The shared worker drives the cells
in cells.py through this backend object; `mode: opt` in the engine options selects the GPU-readout
realization (cells_opt.py).
"""
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend, prompts_of
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class BridgeModel:
    bridge: object             # transformer_lens RemoteBridge over vLLM
    tokenizer: object          # the checkpoint's HF tokenizer, used exactly as the nnsight cells do
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    weights: dict = field(default_factory=dict)   # unembedding and final-norm weight, fetched once
    device: str = "cuda:0"     # where a GPU-side readout projects (the engine shares this process)


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

    def plain(self, model, units, *, new_tokens=1):
        """One forward per prompt (the compiled driver takes one prompt per call) with no captures,
        no interventions and no host-side logit rebuild; the compiled identity hooks still run,
        since they are part of the engine the bridge boots."""
        import torch

        if new_tokens != 1:
            raise NotImplementedError("the vLLM driver runs one forward per call (max_new_tokens=1)")
        for prompt in prompts_of(units):
            model.bridge.run_with_cache(torch.tensor([model.tokenizer(prompt)["input_ids"]]),
                                        names_filter=[], return_logits=False, return_type=None)

    def teardown(self, model) -> None:
        model.bridge.close()


class TransformerLensOptBackend(TransformerLensBackend):
    name = "transformer_lens_opt"


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    mode = params.pop("mode", "default")
    if mode not in ("default", "opt"):
        raise ValueError(f"mode must be default or opt, not {mode!r}")
    if mode == "opt":
        from . import cells_opt  # noqa: F401  (registers the transformer_lens_opt cells)
    cls = TransformerLensOptBackend if mode == "opt" else TransformerLensBackend
    run = RunConfig(EngineConfig("vllm", mode="transformer-lens-opt" if mode == "opt" else "transformer-lens",
                                 params={**params, "enforce_eager": False,
                                         "transformer_lens": metadata.version("transformer-lens")}))
    return cls(**params), run

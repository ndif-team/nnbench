"""The TransformerLens 4.0 vLLM bridge as a system under comparison.

`RemoteBridge.boot_vllm` keeps torch.compile and CUDA graphs: its capture and intervention hooks are
installed before compilation and apply `output * scale + bias`. `enable_batching: true` in the
engine options boots its eager batched path instead (`enforce_eager`, several prompts per forward;
source.py, `boot_vllm`). The shared worker drives the cells in cells.py through this backend object;
`mode: opt` in the engine options selects the GPU-readout realization (cells_opt.py).
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
    weights: dict = field(default_factory=dict)   # unembedding, final norm, J-lens maps: built once
    device: str = "cuda:0"     # where a GPU-side readout projects (the engine shares this process)


class TransformerLensBackend(Backend):
    name = "transformer_lens"

    def __init__(self, dtype: str, **engine_options):
        self.dtype = dtype
        self.engine_options = engine_options
        # the eager batched path takes every unit of a call in one forward (driver.py, forward)
        self.batching = bool(engine_options.get("enable_batching", False))

    def load(self, repo: str):
        import torch
        from transformer_lens.model_bridge import RemoteBridge
        from transformers import AutoConfig, AutoTokenizer

        bridge = RemoteBridge.boot_vllm(repo, dtype=getattr(torch, self.dtype), **self.engine_options)
        config = AutoConfig.from_pretrained(repo)
        return BridgeModel(bridge, AutoTokenizer.from_pretrained(repo), config,
                           config.num_hidden_layers)

    def plain(self, model, units, *, new_tokens=1):
        """One forward per prompt (the compiled driver takes one prompt per call), or one forward
        over every prompt on the batched path, with no captures, no interventions and no host-side
        logit rebuild. The hooks still run, since they are part of the engine the bridge boots: the
        compiled identity affine, or on the batched path a copy of every hooked module's rows to
        host memory (plugin.py, `_make_batched_hook`)."""
        import torch

        if new_tokens != 1:
            raise NotImplementedError("the vLLM driver runs one forward per call (max_new_tokens=1)")
        rows = [model.tokenizer(prompt)["input_ids"] for prompt in prompts_of(units)]
        for ids in [rows] if self.batching else [torch.tensor([row]) for row in rows]:
            model.bridge.run_with_cache(ids, names_filter=[], return_logits=False, return_type=None)

    def collective_rpc(self, model, method, args=()):
        """`LLM.collective_rpc` of the vLLM engine the bridge boots, one result per rank. TL 4.0.0
        exposes no public handle on that engine; this reaches it through the bridge's private
        `_driver` and the driver's private `_llm` (driver.py, `VLLMDriver.__init__`). On one GPU
        boot_vllm keeps the worker in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, source.py),
        so its allocator peak also counts what the cells place on that GPU (the opt readout's
        weights)."""
        return model.bridge._driver._llm.collective_rpc(method, args=args)

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
    # boot_vllm forces enforce_eager on the batched path (source.py) and keeps graphs otherwise
    eager = bool(params.get("enable_batching", False))
    run = RunConfig(EngineConfig("vllm", mode="transformer-lens-opt" if mode == "opt" else "transformer-lens",
                                 params={**params, "enforce_eager": eager,
                                         "transformer_lens": metadata.version("transformer-lens")}))
    return cls(**params), run

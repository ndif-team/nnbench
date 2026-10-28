"""The vLLM-Lens backend: UK AISI's vLLM plugin as a system under comparison.

The shared worker drives the cells in cells.py through this backend object; `mode: opt` in the
engine options selects the fastest documented realization (cells_opt.py). vLLM-Lens registers
through vLLM's general_plugins entry point and forces enforce_eager on every engine.
"""
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend, prompts_of
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class LensModel:
    llm: object
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    dtype: object              # the engine's activation dtype (vLLM's resolved model_config.dtype)
    directions: dict = field(default_factory=dict)   # token id -> unit unembedding row, fetched once
    transports: dict = field(default_factory=dict)   # J-lens transport spec -> maps, built once


class VLLMLensBackend(Backend):
    name = "vllm_lens"

    def __init__(self, **engine_options):
        self.engine_options = engine_options

    def load(self, repo: str):
        import vllm_lens  # noqa: F401  (registers the plugin before the engine is built)
        from transformers import AutoConfig
        from vllm import LLM

        llm = LLM(model=repo, enforce_eager=True, **self.engine_options)
        if self.engine_options.get("tensor_parallel_size", 1) > 1:
            # ctx.get_parameter all-gathers a vocab-sharded lm_head on every call unless it was
            # prefetched (_helpers/types.py, HookContext.get_parameter); the logit-lens and
            # jacobian-lens examples prefetch it once before generating.
            llm.prefetch_params(["lm_head.weight"])
        config = AutoConfig.from_pretrained(repo)
        return LensModel(llm, llm.get_tokenizer(), config, config.num_hidden_layers,
                         llm.llm_engine.model_config.dtype)

    def plain(self, model, units, *, new_tokens=1):
        """A plain generate on the plugin's engine: no extra_args, so no capture, steering or hook."""
        from vllm import SamplingParams

        model.llm.generate(prompts_of(units),
                           SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens,
                                          temperature=0.0, top_p=1.0),
                           use_tqdm=False)

    def collective_rpc(self, model, method, args=()):
        """Through the plugin's `vllm.LLM`, whose `collective_rpc` is vLLM's own (vLLM 0.19.1
        `entrypoints/llm.py`, `LLM.collective_rpc`; vLLM-Lens patches only generate and chat)."""
        return model.llm.collective_rpc(method, args=args)

    def teardown(self, model) -> None:
        model.llm.llm_engine.engine_core.shutdown()


class VLLMLensOptBackend(VLLMLensBackend):
    name = "vllm_lens_opt"


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    mode = params.pop("mode", "default")
    if mode not in ("default", "opt"):
        raise ValueError(f"mode must be default or opt, not {mode!r}")
    if mode == "opt":
        from . import cells_opt  # noqa: F401  (registers the vllm_lens_opt cells)
    cls = VLLMLensOptBackend if mode == "opt" else VLLMLensBackend
    run = RunConfig(EngineConfig("vllm", mode="vllm-lens-opt" if mode == "opt" else "vllm-lens", params={
        **params, "enforce_eager": True, "vllm_lens": metadata.version("vllm-lens")}))
    return cls(**params), run

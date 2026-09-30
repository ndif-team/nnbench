"""The vLLM-Lens backend: UK AISI's vLLM plugin as a system under comparison (design.md §12.14).

The shared worker drives the cells in cells.py through this backend object. vLLM-Lens registers
through vLLM's general_plugins entry point and forces enforce_eager on every engine.
"""
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class LensModel:
    llm: object
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    directions: dict = field(default_factory=dict)   # token id -> unit unembedding row, fetched once


class VLLMLensBackend(Backend):
    name = "vllm_lens"

    def __init__(self, **engine_options):
        self.engine_options = engine_options

    def load(self, repo: str):
        import vllm_lens  # noqa: F401  (registers the plugin before the engine is built)
        from transformers import AutoConfig
        from vllm import LLM

        llm = LLM(model=repo, enforce_eager=True, **self.engine_options)
        config = AutoConfig.from_pretrained(repo)
        return LensModel(llm, llm.get_tokenizer(), config, config.num_hidden_layers)

    def vanilla(self, model, prompt, *, new_tokens):
        """A plain generate on the plugin's engine: no extra_args, so no capture, steering or hook."""
        from vllm import SamplingParams

        model.llm.generate([prompt], SamplingParams(max_tokens=new_tokens, temperature=0.0, top_p=1.0),
                           use_tqdm=False)

    def teardown(self, model) -> None:
        model.llm.llm_engine.engine_core.shutdown()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    import cells  # noqa: F401  (registers the vllm_lens cells)

    run = RunConfig(EngineConfig("vllm", mode="vllm-lens", params={
        **params, "enforce_eager": True, "vllm_lens": metadata.version("vllm-lens")}))
    return VLLMLensBackend(**params), run


if __name__ == "__main__":
    main(create_backend)

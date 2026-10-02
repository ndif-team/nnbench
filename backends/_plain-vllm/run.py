"""Plain vLLM with no interpretability layer: the whole-system denominator (design.md §12.14).

Shared by the vllm-plain-* backends, one per engine version and execution mode. It times only the
no-intervention request; every intervention cell is declared unsupported. `vllm-plain-X` keeps
vLLM's default configuration (torch.compile and CUDA graphs), which is what a user gets from plain
vLLM; `vllm-plain-eager-X` sets enforce_eager, the mode the eager-only systems run in, so their
overhead is not confounded with the CUDA-graph gap (`isb/jobs/score.py`, `attach_plain_vllm`).
"""
from dataclasses import dataclass

from isb.backends.base import Backend
from isb.jobs.worker import main, merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class PlainModel:
    llm: object
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision


class PlainVLLMBackend(Backend):
    name = "vllm_plain"

    def __init__(self, **engine_options):
        self.engine_options = engine_options

    def load(self, repo: str):
        from transformers import AutoConfig
        from vllm import LLM

        llm = LLM(model=repo, **self.engine_options)
        return PlainModel(llm, llm.get_tokenizer(), AutoConfig.from_pretrained(repo))

    def vanilla(self, model, prompt, *, new_tokens):
        from vllm import SamplingParams

        model.llm.generate([prompt], SamplingParams(max_tokens=new_tokens, temperature=0.0, top_p=1.0),
                           use_tqdm=False)

    def teardown(self, model) -> None:
        model.llm.llm_engine.engine_core.shutdown()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    import cells  # noqa: F401  (declares every intervention unsupported)

    run = RunConfig(EngineConfig("vllm", mode="plain", params={
        **params, "enforce_eager": bool(params.get("enforce_eager", False))}))
    return PlainVLLMBackend(**params), run


if __name__ == "__main__":
    main(create_backend)

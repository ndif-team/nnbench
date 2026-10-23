"""Plain vLLM served through AsyncLLM, the entrypoint `vllm serve` uses: the floor under
concurrent and open-loop load (design.md §12.17). `vllm-plain/<version>` builds the offline `LLM`
class, which takes one call at a time; this config takes overlapping requests on the engine's own
scheduler, so a plain row under load measures vLLM's serving path and nothing in front of it.
Every intervention cell is declared unsupported, as in the sync plain backend.
"""
from dataclasses import dataclass

from isb.backends.async_loop import AsyncLoopMixin
from isb.backends.base import Backend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class PlainModel:
    llm: object                # vllm.v1.engine.async_llm.AsyncLLM
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision


class PlainAsyncVLLMBackend(AsyncLoopMixin, Backend):
    name = "vllm_plain"

    def __init__(self, *, enforce_eager, **engine_options):
        # Every config states the mode (compose.base.yml): attach_plain_vllm pairs baselines on it.
        self.engine_options = {**engine_options, "enforce_eager": enforce_eager}

    def load(self, repo: str):
        from transformers import AutoConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model=repo, **self.engine_options))
        tokenizer = self._run_coro(_tokenizer(llm))
        return PlainModel(llm, tokenizer, AutoConfig.from_pretrained(repo))

    def vanilla(self, model, prompt, *, new_tokens):
        """One request straight to AsyncLLM.generate, no intervention."""
        self._run_coro(_one(model.llm, prompt, new_tokens))

    def vanilla_many(self, model, prompts, *, new_tokens):
        """`vanilla` over several prompts submitted together, which the engine batches."""
        import asyncio

        async def _go():
            await asyncio.gather(*(_one(model.llm, p, new_tokens) for p in prompts))
        self._run_coro(_go())

    def teardown(self, model) -> None:
        model.llm.shutdown()
        self._close_loop()


async def _tokenizer(llm):
    getter = llm.get_tokenizer
    result = getter()
    return await result if hasattr(result, "__await__") else result


async def _one(llm, prompt, new_tokens):
    import uuid

    from vllm import SamplingParams

    params = SamplingParams(max_tokens=new_tokens, temperature=0.0, top_p=1.0)
    async for _ in llm.generate(prompt, params, uuid.uuid4().hex):
        pass


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    backend = PlainAsyncVLLMBackend(**params)
    return backend, RunConfig(EngineConfig("vllm", mode="plain", params={
        **backend.engine_options, "entrypoint": "AsyncLLM"}))

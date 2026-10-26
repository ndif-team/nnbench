"""Plain vLLM with no interpretability layer: the whole-system denominator.

Shared by every vllm-plain config (configs.yml), one per engine version and execution mode. It
times only the no-intervention request; every intervention cell is declared unsupported.
`vllm-plain/<version>` keeps vLLM's default configuration (torch.compile and CUDA graphs), which is
what a user gets from plain vLLM; `vllm-plain/<version>/eager` sets enforce_eager, the mode the
eager-only systems run in, so their overhead is not confounded with the CUDA-graph gap
(`isb/jobs/score.py`, `attach_plain_vllm`). The engine is vLLM's serving engine (AsyncLLM), so
concurrent and open-loop clients are batched together as a vLLM server batches them.
"""
import contextlib
from dataclasses import dataclass

from isb.backends.base import Backend, prompts_of
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


@dataclass
class PlainModel:
    llm: object                # vllm.v1.engine.async_llm.AsyncLLM
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision


class PlainVLLMBackend(Backend):
    name = "vllm_plain"
    # Several client threads may submit at once: AsyncLLM is a serving engine.
    concurrent_requests = True

    def __init__(self, *, enforce_eager, **engine_options):
        # Every config states the mode: attach_plain_vllm pairs baselines on it. Stats logging is
        # off unless a config asks for it, as `vllm.LLM` sets it.
        self.engine_options = {"disable_log_stats": True, **engine_options,
                               "enforce_eager": enforce_eager}
        self._loop = None
        self._loop_thread = None         # set while a concurrent session runs the loop

    def load(self, repo: str):
        from transformers import AutoConfig, AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        # LLM_CLASS: the scheduler defaults `vllm.LLM` derives (max_num_batched_tokens, max_num_seqs)
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model=repo, **self.engine_options),
                                        usage_context=UsageContext.LLM_CLASS)
        trust_remote_code = self.engine_options.get("trust_remote_code", False)
        return PlainModel(llm, AutoTokenizer.from_pretrained(repo, trust_remote_code=trust_remote_code),
                          AutoConfig.from_pretrained(repo, trust_remote_code=trust_remote_code))

    def _gather(self, coros):
        """Run the coroutines together on the one persistent loop (a fresh loop per call would close
        the engine's output handler): on this thread, or on a concurrent session's loop thread."""
        import asyncio

        async def every():
            return await asyncio.gather(*coros)

        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        if self._loop_thread is not None:
            return asyncio.run_coroutine_threadsafe(every(), self._loop).result()
        return self._loop.run_until_complete(every())

    @contextlib.contextmanager
    def concurrent_session(self):
        """Run the persistent loop on its own thread while clients submit from theirs."""
        import asyncio
        import threading

        if self._loop_thread is not None:
            raise RuntimeError("a concurrent session is already open")
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        loop = self._loop
        thread = threading.Thread(target=loop.run_forever, name="isb-vllm-plain-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        try:
            yield
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._loop_thread = None

    def plain(self, model, units, *, new_tokens=1):
        """One request per prompt, submitted together: the engine batches them as `LLM.generate`
        batches a list."""
        import uuid

        from vllm import SamplingParams

        params = SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0,
                                top_p=1.0)

        async def one(prompt):
            async for _ in model.llm.generate(prompt, params, uuid.uuid4().hex):
                pass

        self._gather([one(prompt) for prompt in prompts_of(units)])

    def teardown(self, model) -> None:
        import asyncio

        model.llm.shutdown()
        loop, self._loop = self._loop, None
        if loop is None:
            return

        async def drain():                       # let the engine's cancelled tasks finish
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        loop.run_until_complete(drain())
        loop.close()


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    backend = PlainVLLMBackend(**params)
    return backend, RunConfig(EngineConfig("vllm", mode="plain", params=backend.engine_options))

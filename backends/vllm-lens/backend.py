"""The vLLM-Lens backend: UK AISI's vLLM plugin as a system under comparison.

The shared worker drives the cells in cells.py through this backend object; `mode: opt` in the
engine options selects the fastest documented realization (cells_opt.py), and `async_llm: true`
runs the same cells on vLLM's serving engine, AsyncLLM, one request per prompt (`_AsyncLLMEngine`).
vLLM-Lens registers through vLLM's general_plugins entry point and forces enforce_eager on every
engine.
"""
import contextlib
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


class SerialTokenizer:
    """The clients' own tokenizer, one call at a time. A Hugging Face fast tokenizer raises
    "Already borrowed" when two threads call it at once (each call sets its truncation state), so
    concurrent clients cannot share one instance with each other or with the engine's renderer,
    which tokenizes every prompt with its own (vLLM 0.19.1 `renderers/base.py`, `_tokenize_prompt`).
    Every method call goes through one lock."""

    def __init__(self, tokenizer):
        import threading

        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._lock:
            return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self._tokenizer, name)
        if not callable(attr):
            return attr

        def serial(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)
        return serial


class AsyncRequests:
    """`LLM.generate`'s call form over an AsyncLLM, so the cells run unchanged: each prompt is its
    own request under a fresh id, all submitted together on the backend's persistent loop, and the
    finished outputs return in prompt order. vLLM-Lens's AsyncLLM patch attaches `hook_results` and
    `activations` to each finished output (`_activations_plugin.py`, `_patched_generate`)."""

    def __init__(self, engine, gather):
        self.engine = engine                 # vllm.v1.engine.async_llm.AsyncLLM
        self._gather = gather

    def generate(self, prompts, params, use_tqdm=False):
        import uuid

        # The patch pops `apply_hooks` and `apply_steering_vectors` off the request's own
        # SamplingParams, so a list-wide params object is copied per request.
        if not isinstance(params, list):
            params = [params.clone() for _ in prompts]

        async def one(prompt, request_params):
            async for out in self.engine.generate(prompt, request_params, uuid.uuid4().hex):
                pass
            return out

        return self._gather([one(prompt, p) for prompt, p in zip(prompts, params)])


class _AsyncLLMEngine:
    """vLLM-Lens on vLLM's serving engine, AsyncLLM, in-process: the plugin patches
    `AsyncLLM.generate` for online use (`_activations_plugin.py`, module docstring and
    `_patched_generate`; its tests build `AsyncLLMEngine.from_engine_args`, an alias of AsyncLLM in
    vLLM 0.19.1 `engine/async_llm_engine.py`). The worker matches a request's hooks, steering
    vectors and captures by its external id followed by vLLM's random suffix (`_worker_ext.py`,
    `get_hook_results`; vLLM 0.19.1 `v1/engine/input_processor.py`, `assign_request_id`).
    Concurrent and open-loop clients submit to one engine, which batches them, as a vLLM server
    does. The persistent loop and its session follow backends/vllm-plain/backend.py."""
    # Several client threads may submit at once: AsyncLLM is a serving engine.
    concurrent_requests = True

    def __init__(self, **engine_options):
        # Stats logging is off unless a config asks for it, as `vllm.LLM` and vllm-plain set it.
        super().__init__(**{"disable_log_stats": True, **engine_options})
        self._loop = None
        self._loop_thread = None         # set while a concurrent session runs the loop

    def load(self, repo: str):
        from transformers import AutoConfig, AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM

        # AsyncEngineArgs loads the general plugins (vLLM 0.19.1 `engine/arg_utils.py`,
        # `EngineArgs.__post_init__`), so vLLM-Lens patches the engine before it is built.
        # LLM_CLASS: the scheduler defaults `vllm.LLM` derives, as on vllm-plain.
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(model=repo, enforce_eager=True, **self.engine_options),
            usage_context=UsageContext.LLM_CLASS)
        if self.engine_options.get("tensor_parallel_size", 1) > 1:
            # The RPC `LLM.prefetch_params` sends (`_activations_plugin.py`,
            # `_llm_prefetch_params`); AsyncLLM has no such method.
            self._gather([engine.collective_rpc("prefetch_parameters", args=(["lm_head.weight"],))])
        config = AutoConfig.from_pretrained(repo)
        return LensModel(AsyncRequests(engine, self._gather),
                         SerialTokenizer(AutoTokenizer.from_pretrained(repo)), config,
                         config.num_hidden_layers, engine.model_config.dtype)

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
        thread = threading.Thread(target=loop.run_forever, name="isb-vllm-lens-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        try:
            yield
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._loop_thread = None

    def plain(self, model, units, *, new_tokens=1):
        """No extra_args, so the patch installs and sends nothing for the request."""
        from vllm import SamplingParams

        model.llm.generate(prompts_of(units),
                           SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens,
                                          temperature=0.0, top_p=1.0))

    def collective_rpc(self, model, method, args=()):
        """Through the AsyncLLM's own `collective_rpc`, on the persistent loop."""
        return self._gather([model.llm.engine.collective_rpc(method, args=args)])[0]

    def teardown(self, model) -> None:
        import asyncio

        model.llm.engine.shutdown()
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


class VLLMLensAsyncBackend(_AsyncLLMEngine, VLLMLensBackend):
    pass


class VLLMLensOptAsyncBackend(_AsyncLLMEngine, VLLMLensOptBackend):
    pass


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    mode = params.pop("mode", "default")
    if mode not in ("default", "opt"):
        raise ValueError(f"mode must be default or opt, not {mode!r}")
    # async_llm: the engine is vLLM's AsyncLLM, one request per prompt, instead of `vllm.LLM`.
    async_llm = params.pop("async_llm", False)
    if mode == "opt":
        from . import cells_opt  # noqa: F401  (registers the vllm_lens_opt cells)
    cls = {("default", False): VLLMLensBackend, ("opt", False): VLLMLensOptBackend,
           ("default", True): VLLMLensAsyncBackend, ("opt", True): VLLMLensOptAsyncBackend}
    backend = cls[mode, async_llm](**params)
    name = "vllm-lens-opt" if mode == "opt" else "vllm-lens"
    run = RunConfig(EngineConfig("vllm", mode=f"{name}-async" if async_llm else name, params={
        **backend.engine_options, "enforce_eager": True,
        "vllm_lens": metadata.version("vllm-lens")}))
    return backend, run

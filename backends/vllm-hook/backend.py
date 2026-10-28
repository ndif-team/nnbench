"""The vLLM-Hook backend: IBM's vLLM plugin as a system under comparison.

Driven through its documented offline entry point, `HookLLM` (README, docs/configs.md), with a
built-in worker and its analyzer (docs/configs.md, "Selecting a configuration from user code"):
`probe_hidden_states` with `hidden_states` (the default), or `probe_hook_qk` with `attn_tracker`
for the attention cells. The plugin registers through vLLM's general_plugins entry point and
forces enforce_eager on every engine (`_hook_plugin._patched_create_engine_config`). An engine
carries one worker extension, so one engine reads hidden states or Q/K, and neither steers
(`steer_hook_act` is a different worker).

Storage is the documented per-request axis (docs/configs.md): "rpc" returns probes on the output
(`HookLLM.generate` default); "disk-st-async" writes a safetensors artifact from a background
thread, the variant the docs recommend as fastest. Both read back through `HookLLM.analyze`.
`async_llm: true` runs the hidden-state cells on vLLM's serving engine, AsyncLLM, one request per
prompt with its own extra_args (`VLLMHookAsyncBackend`, `AsyncHook`).
"""
import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend, prompts_of
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig

WORKERS = {  # docs/configs.md: worker_name -> its analyzer_name
    "probe_hidden_states": "hidden_states",
    "probe_hook_qk": "attn_tracker",
}
STORAGE_ENV = {  # docs/configs.md: set before HookLLM is built; the worker inherits them at spawn
    "rpc": {},
    "disk-st-async": {"VLLM_HOOK_USE_SAFETENSORS": "1", "VLLM_HOOK_ASYNC_SAVE": "1"},
}


@dataclass
class HookModel:
    llm: object                # vllm_hook_plugins.HookLLM
    tokenizer: object
    config: object             # the checkpoint's HF config; model identity reads its revision
    n_layers: int
    repo: str
    save_to_disk: bool
    worker: str                # the engine's worker extension (WORKERS)
    weights: dict = field(default_factory=dict)   # final norm + unembedding from the checkpoint
    transports: dict = field(default_factory=dict)   # J-lens transport spec -> maps, built once


class VLLMHookBackend(Backend):
    name = "vllm_hook"

    def __init__(self, storage="rpc", worker="probe_hidden_states", **engine_options):
        if storage not in STORAGE_ENV:
            raise ValueError(f"unknown vLLM-Hook storage {storage!r}; one of {sorted(STORAGE_ENV)}")
        if worker not in WORKERS:
            raise ValueError(f"unknown vLLM-Hook worker {worker!r}; one of {sorted(WORKERS)}")
        self.storage = storage
        self.worker = worker
        self.engine_options = engine_options
        self.hook_dir = None

    def load(self, repo: str):
        os.environ.update(STORAGE_ENV[self.storage])
        from transformers import AutoConfig
        from vllm_hook_plugins import HookLLM

        # /dev/shm is the plugin's default artifact root (_hook_plugin._DEFAULT_HOOK_DIR).
        self.hook_dir = tempfile.mkdtemp(prefix="isb_vllm_hook_", dir="/dev/shm")
        config = AutoConfig.from_pretrained(repo)
        config_file = None
        if self.worker == "probe_hook_qk":
            # A model config in the form of model_configs/attention_tracker/*.json: every (layer,
            # head) pair as an important head, so the attn_tracker analyzer reconstructs every
            # head (attention_tracker_analyzer.compute_attention_from_qk), and the query captured
            # at the last token only (hookq_mode, probe_hookqk_worker.qkv_hook).
            config_file = os.path.join(self.hook_dir, "attention.json")
            heads = [[layer, head] for layer in range(config.num_hidden_layers)
                     for head in range(config.num_attention_heads)]
            with open(config_file, "w") as f:
                json.dump({"params": {"important_heads": heads},
                           "hookq": {"hookq_mode": "last_token"}}, f)
        # download_dir=None: vLLM resolves the model through HF_HOME (HookLLM's default is ~/.cache).
        llm = HookLLM(model=repo, worker_name=self.worker, analyzer_name=WORKERS[self.worker],
                      config_file=config_file, download_dir=None, hook_dir=self.hook_dir,
                      enforce_eager=True, **self.engine_options)
        llm._hs_mode = "last_token"     # what load_config sets from a config's hidden_states.mode
        return HookModel(llm, llm.tokenizer, config, config.num_hidden_layers, repo,
                         save_to_disk=self.storage != "rpc", worker=self.worker)

    def plain(self, model, units, *, new_tokens=1):
        """A plain generate on the plugin's engine: use_hook=False clears extra_args."""
        from vllm import SamplingParams

        model.llm.generate(prompts_of(units),
                           SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens,
                                          temperature=0.0, top_p=1.0),
                           use_hook=False, use_tqdm=False)

    def collective_rpc(self, model, method, args=()):
        """Through `HookLLM.llm`, the `vllm.LLM` it wraps (hook_llm.py, `HookLLM.__init__`), whose
        `collective_rpc` is vLLM's own (vLLM 0.19.1 `entrypoints/llm.py`, `LLM.collective_rpc`)."""
        return model.llm.llm.collective_rpc(method, args=args)

    def teardown(self, model) -> None:
        model.llm.llm_engine.engine_core.shutdown()
        model.llm.close()
        if self.hook_dir:
            shutil.rmtree(self.hook_dir, ignore_errors=True)


@dataclass
class AsyncHook:
    """The plugin's online path, AsyncLLM in-process: `_hook_plugin._patched_generate` reads each
    request's own extra_args (docs/configs.md, "All hook activation is per-request"), returns the
    probes on the finished output (rpc), or awaits the worker's `flush_disk` of that request under
    its run_id (disk). The fields are the ones HookLLM and HookClient send (`hook_llm.py`,
    `_build_extra_args`; `hook_client.py`, `generate`), and the artifacts read back through the same
    analyzer `analyze` call (`run_utils.dispatch_disk_analyze`)."""
    engine: object             # vllm.v1.engine.async_llm.AsyncLLM
    analyzer: object           # the worker's analyzer, built as HookLLM builds it
    hook_dir: str
    save_to_disk: bool
    gather: object             # the backend's persistent-loop runner

    def capture(self, prompts, layers, mode):
        """One request per prompt, submitted together, capturing the stream after each listed
        layer (numbered from 1, as the worker numbers them) in hs mode `mode`; returns the finished
        outputs and `{module: rows}` with every request's passes in prompt order. The artifacts
        are read on the calling client's thread, off the engine's event loop."""
        import uuid

        from vllm import SamplingParams

        ids = [uuid.uuid4().hex for _ in prompts]

        def params(request_id):
            extra = {"output_hidden_states": list(layers), "hs_mode": mode}
            if self.save_to_disk:     # the request id names its artifact directory
                extra.update(save_to_disk=True, run_id=request_id, hook_dir=self.hook_dir)
            return SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, extra_args=extra)

        async def one(prompt, request_id):
            async for out in self.engine.generate(prompt, params(request_id), request_id):
                pass
            return out

        outs = self.gather([one(prompt, request_id) for prompt, request_id in zip(prompts, ids)])
        by_layer = {}
        for out, request_id in zip(outs, ids):
            if self.save_to_disk:
                stats = self.analyzer.analyze(analyzer_spec={"reduce": "none"}, run_id=request_id)
                shutil.rmtree(os.path.join(self.hook_dir, request_id), ignore_errors=True)
            else:
                stats = self.analyzer.analyze(analyzer_spec={"reduce": "none"}, probes=out.probes)
            for module, rows in stats["hidden_states"].items():
                by_layer.setdefault(module, []).extend(rows)
        return outs, by_layer


class VLLMHookAsyncBackend(VLLMHookBackend):
    """vLLM-Hook on vLLM's serving engine, AsyncLLM, in-process: the engine `vllm serve` runs
    behind the plugin's documented serve path (docs/configs.md, "Serve"), driven here without the
    HTTP server. Concurrent and open-loop clients submit to one engine, which batches them, as a
    vLLM server does. The persistent loop and its session follow backends/vllm-plain/backend.py.
    Cells run through `AsyncHook.capture`."""
    name = "vllm_hook_async"
    # Several client threads may submit at once: AsyncLLM is a serving engine.
    concurrent_requests = True

    def __init__(self, **options):
        super().__init__(**options)
        # Stats logging is off unless a config asks for it, as `vllm.LLM` and vllm-plain set it.
        self.engine_options = {"disable_log_stats": True, **self.engine_options}
        self._loop = None
        self._loop_thread = None         # set while a concurrent session runs the loop

    def load(self, repo: str):
        if self.worker != "probe_hidden_states":
            raise ValueError("the AsyncLLM backend reads hidden states only (AsyncHook.capture)")
        os.environ.update(STORAGE_ENV[self.storage])     # the engine-core process inherits them
        import vllm.plugins
        from transformers import AutoConfig, AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm_hook_plugins.registry import PluginRegistry

        self.hook_dir = tempfile.mkdtemp(prefix="isb_vllm_hook_", dir="/dev/shm")
        # The worker extension and analyzer as HookLLM resolves them (`hook_llm.py`, __init__).
        vllm.plugins.load_general_plugins()
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(model=repo, enforce_eager=True,
                            worker_extension_cls=PluginRegistry.get_worker(self.worker).path,
                            **self.engine_options),
            usage_context=UsageContext.LLM_CLASS)    # vllm.LLM's scheduler defaults, as vllm-plain
        analyzer = PluginRegistry.get_analyzer(WORKERS[self.worker]).analyzer(self.hook_dir, {})
        config = AutoConfig.from_pretrained(repo)
        llm = AsyncHook(engine, analyzer, self.hook_dir, self.storage != "rpc", self._gather)
        # The clients' own tokenizer: the engine's renderer tokenizes with its instance, and a
        # fast tokenizer raises "Already borrowed" under concurrent calls.
        return HookModel(llm, AutoTokenizer.from_pretrained(repo), config,
                         config.num_hidden_layers, repo, save_to_disk=llm.save_to_disk,
                         worker=self.worker)

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
        thread = threading.Thread(target=loop.run_forever, name="isb-vllm-hook-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        try:
            yield
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._loop_thread = None

    def plain(self, model, units, *, new_tokens=1):
        """One request per prompt with no extra_args, so the patch installs and sends nothing."""
        import uuid

        from vllm import SamplingParams

        params = SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0,
                                top_p=1.0)

        async def one(prompt):
            async for _ in model.llm.engine.generate(prompt, params, uuid.uuid4().hex):
                pass

        self._gather([one(prompt) for prompt in prompts_of(units)])

    def collective_rpc(self, model, method, args=()):
        """Through the AsyncLLM's own `collective_rpc`, on the persistent loop."""
        return self._gather([model.llm.engine.collective_rpc(method, args=args)])[0]

    def teardown(self, model) -> None:
        import asyncio

        model.llm.engine.shutdown()
        if self.hook_dir:
            shutil.rmtree(self.hook_dir, ignore_errors=True)
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
    storage = params.pop("storage", "rpc")
    worker = params.pop("worker", "probe_hidden_states")
    # async_llm: the engine is vLLM's AsyncLLM, one request per prompt, instead of HookLLM.
    async_llm = params.pop("async_llm", False)
    backend = (VLLMHookAsyncBackend if async_llm else VLLMHookBackend)(
        storage=storage, worker=worker, **params)
    mode = f"vllm-hook-{storage}-async" if async_llm else f"vllm-hook-{storage}"
    run = RunConfig(EngineConfig("vllm", mode=mode, params={
        **backend.engine_options, "enforce_eager": True, "storage": storage, "worker": worker,
        "vllm_hook_plugins": metadata.version("vllm-hook-plugins")}))
    return backend, run

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
"""
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


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    storage = params.pop("storage", "rpc")
    worker = params.pop("worker", "probe_hidden_states")
    run = RunConfig(EngineConfig("vllm", mode=f"vllm-hook-{storage}", params={
        **params, "enforce_eager": True, "storage": storage, "worker": worker,
        "vllm_hook_plugins": metadata.version("vllm-hook-plugins")}))
    return VLLMHookBackend(storage=storage, worker=worker, **params), run

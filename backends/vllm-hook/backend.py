"""The vLLM-Hook backend: IBM's vLLM plugin as a system under comparison (design.md §12.14).

Driven through its documented offline entry point, `HookLLM` (README, docs/configs.md), with the
built-in `probe_hidden_states` worker and `hidden_states` analyzer. The plugin registers through
vLLM's general_plugins entry point and forces enforce_eager on every engine
(`_hook_plugin._patched_create_engine_config`). An engine carries one worker extension, so the one
loaded here reads hidden states and cannot steer (`steer_hook_act` is a different worker).

Storage is the documented per-request axis (docs/configs.md): "rpc" returns probes on the output
(`HookLLM.generate` default); "disk-st-async" writes a safetensors artifact from a background
thread, the variant the docs recommend as fastest. Both read back through `HookLLM.analyze`.
"""
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from importlib import metadata

from isb.backends.base import Backend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig

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
    weights: dict = field(default_factory=dict)   # final norm + unembedding from the checkpoint


class VLLMHookBackend(Backend):
    name = "vllm_hook"

    def __init__(self, storage="rpc", **engine_options):
        if storage not in STORAGE_ENV:
            raise ValueError(f"unknown vLLM-Hook storage {storage!r}; one of {sorted(STORAGE_ENV)}")
        self.storage = storage
        self.engine_options = engine_options
        self.hook_dir = None

    def load(self, repo: str):
        os.environ.update(STORAGE_ENV[self.storage])
        from transformers import AutoConfig
        from vllm_hook_plugins import HookLLM

        # /dev/shm is the plugin's default artifact root (_hook_plugin._DEFAULT_HOOK_DIR).
        self.hook_dir = tempfile.mkdtemp(prefix="isb_vllm_hook_", dir="/dev/shm")
        # download_dir=None: vLLM resolves the model through HF_HOME (HookLLM's default is ~/.cache).
        llm = HookLLM(model=repo, worker_name="probe_hidden_states", analyzer_name="hidden_states",
                      download_dir=None, hook_dir=self.hook_dir, enforce_eager=True,
                      **self.engine_options)
        llm._hs_mode = "last_token"     # what load_config sets from a config's hidden_states.mode
        config = AutoConfig.from_pretrained(repo)
        return HookModel(llm, llm.tokenizer, config, config.num_hidden_layers, repo,
                         save_to_disk=self.storage != "rpc")

    def vanilla(self, model, prompt, *, new_tokens):
        """A plain generate on the plugin's engine: use_hook=False clears extra_args."""
        from vllm import SamplingParams

        model.llm.generate([prompt], SamplingParams(max_tokens=new_tokens, temperature=0.0, top_p=1.0),
                           use_hook=False, use_tqdm=False)

    def teardown(self, model) -> None:
        model.llm.llm_engine.engine_core.shutdown()
        model.llm.close()
        if self.hook_dir:
            shutil.rmtree(self.hook_dir, ignore_errors=True)


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    storage = params.pop("storage", "rpc")
    run = RunConfig(EngineConfig("vllm", mode=f"vllm-hook-{storage}", params={
        **params, "enforce_eager": True, "storage": storage,
        "vllm_hook_plugins": metadata.version("vllm-hook-plugins")}))
    return VLLMHookBackend(storage=storage, **params), run

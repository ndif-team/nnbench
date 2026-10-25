# One backend per process: a vLLM container imports its engine at load, before any trace. An HF
# trace run first mounts nnsight's `save` onto `object`, and anyio's attribute check then fails for
# the classes vLLM's import defines afterwards.
from .base import Backend
from .hf import HFBackend
from .vllm_async import VLLMAsyncBackend
from .vllm_serve import VLLMServeBackend
from .vllm_sync import VLLMSyncBackend

IMPLS = {
    HFBackend.name: HFBackend,
    VLLMAsyncBackend.name: VLLMAsyncBackend,
    VLLMSyncBackend.name: VLLMSyncBackend,
    VLLMServeBackend.name: VLLMServeBackend,
}

__all__ = [
    "Backend", "HFBackend", "VLLMAsyncBackend", "VLLMSyncBackend",
    "VLLMServeBackend", "IMPLS",
]

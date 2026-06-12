from .base import Backend
from .hf import HFBackend
from .vllm_async import VLLMAsyncBackend
from .vllm_serve import VLLMServeBackend

IMPLS = {
    HFBackend.name: HFBackend,
    VLLMAsyncBackend.name: VLLMAsyncBackend,
    VLLMServeBackend.name: VLLMServeBackend,
}

__all__ = ["Backend", "HFBackend", "VLLMAsyncBackend", "VLLMServeBackend", "IMPLS"]

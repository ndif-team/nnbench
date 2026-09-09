# NOTE (nnsight-side finding, measured 2026-07-22 on the pp-on-dev editable): running any trace
# permanently mounts `save` onto `object` (globals.py PYMOUNT), after which anyio's strict
# TypedAttributeSet check fails for any anyio class DEFINED later — so a process must never run an
# HF trace before its first `import vllm` (-> openai -> anyio). The split-run design (§12.9) makes
# that order impossible from the CLI: every container runs ONE backend, and a vLLM container
# imports its engine at load time, before any trace. If a caller runs an HF trace before creating
# a vLLM backend in ONE process, this failure mode returns.
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

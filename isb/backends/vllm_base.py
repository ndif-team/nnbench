"""Shared engine-config base for the vLLM backends (design.md §12.3).

The three vLLM backends — in-process async, in-process sync, and over-HTTP serve — differ only in
HOW they run a trace (event loop / in-frame / wire). They share the same vLLM *engine config*: the
precision `dtype`, and `trust_remote_code` (which permits vLLM to READ an auto_map repo's config,
e.g. NemotronH — the native impl runs, no remote modeling is executed). The driver splats a spec's
`vllm_kwargs` into ANY of their constructors, so they must accept ONE signature for that config —
defining it here means a new shared knob can't be added to one backend and silently rejected by the
others (the inconsistency this base exists to prevent).
"""
from __future__ import annotations

from .base import Backend


class VLLMBackend(Backend):
    def __init__(self, dtype: str | None = None, trust_remote_code: bool = False):
        # dtype: precision axis (None -> vLLM's default, bf16 for GPT-2; "float32" matches HF, which is
        # how the oracle separates SUPPORTED_DEGRADED from a true SILENTLY_WRONG bug).
        self.dtype = dtype
        # trust_remote_code: lets vLLM read a repo whose config declares custom code (auto_map). The
        # remote modeling is never executed (vLLM uses its native impl) — needed by NemotronH, arrives
        # via spec.vllm_kwargs.
        self.trust_remote_code = trust_remote_code

    def _engine_kwargs(self) -> dict:
        """vLLM engine kwargs shared by the in-process backends — fed to nnsight's `VLLM(...)`."""
        kw: dict = {}
        if self.dtype is not None:
            kw["dtype"] = self.dtype
        if self.trust_remote_code:
            kw["trust_remote_code"] = True
        return kw

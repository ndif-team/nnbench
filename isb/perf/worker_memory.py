"""Peak GPU memory inside a vLLM engine's worker processes.

vLLM V1 runs the model in separate engine processes, so `torch.cuda.max_memory_allocated()` in the
benchmark process sees only what reaches the client. The engine's own `collective_rpc` (`LLM` and
`AsyncLLM`, vLLM 0.19.1 to 0.28.0) runs a callable on every worker and returns one result per
rank; the two functions below reset and read the PyTorch allocator's peak there. A callable crosses
the process boundary only with `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (vllm/v1/serial_utils.py),
which `allow_callable_rpc` sets before the engine starts; the engine processes inherit it.
"""
from __future__ import annotations

import os

INSECURE_SERIALIZATION = "VLLM_ALLOW_INSECURE_SERIALIZATION"


def allow_callable_rpc() -> None:
    """Let `collective_rpc` carry a Python callable to the engine processes started after this."""
    os.environ[INSECURE_SERIALIZATION] = "1"


def reset_peak(worker) -> None:
    """Runs in each engine worker: start a new peak from the memory allocated now."""
    import torch

    torch.cuda.reset_peak_memory_stats()


def max_allocated(worker) -> int:
    """Runs in each engine worker: the allocator's peak since the last reset, in bytes."""
    import torch

    return torch.cuda.max_memory_allocated()


class WorkerMemory:
    """The worker peak of one measured call series on backend `be`, through `be.collective_rpc`.
    A backend with no engine workers to reach raises NotImplementedError there; that reason, or any
    other error the RPC raises, is kept in `record()` instead of failing the measured cell."""

    def __init__(self, be, model):
        self.be, self.model = be, model
        self.unavailable = self.error = None

    def _rpc(self, fn):
        if self.unavailable is not None or self.error is not None:
            return None
        try:
            return self.be.collective_rpc(self.model, fn)
        except NotImplementedError as reason:
            self.unavailable = str(reason)
        except Exception as error:
            self.error = repr(error)[:500]
        return None

    def reset(self) -> None:
        self._rpc(reset_peak)

    def peak_mb(self):
        """Per-rank peak in MiB since `reset`, or None when it could not be read."""
        peaks = self._rpc(max_allocated)
        return None if peaks is None else [round(b / 2 ** 20, 1) for b in peaks]

    def record(self, peaks) -> dict:
        """The fields a call record carries for the per-rank `peaks` (None when unread)."""
        if peaks is not None:
            return {"worker_peak_mem_mb": peaks, "worker_peak_mem_mb_max": max(peaks)}
        if self.unavailable is not None:
            return {"worker_peak_mem_mb": None, "worker_memory_unavailable": self.unavailable}
        return {"worker_peak_mem_mb": None, "worker_memory_error": self.error}

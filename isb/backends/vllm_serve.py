"""vLLM serve-client backend — the VM-style, over-the-wire counterpart to `vllm_async`.

Where `vllm_async` runs the vLLM engine IN-PROCESS, this backend builds only a META `VLLM` model (no
engine, no GPU — the CPU platform when NVML sees zero devices, e.g. a GPU-less container) and submits
the compiled trace to a SEPARATE nnsight-vllm-serve instance over HTTP via `model.trace(serve=url)`.
The server runs the intervention on its real vLLM engine, collects the `.save()`d values, and returns
them; nnsight's `tracer.push` (blocking mode) injects them back into this frame as CPU tensors.

This is structurally SIMPLER than the async backend: the blocking serve path has no event loop, no
output stream, and no `.saves` envelope walk. Worker-side intervention errors surface here as raised
`RuntimeError`s (`local_serve.surface_server_errors`), and saved values arrive already materialized on
CPU (`torch.load(map_location="cpu")`). The cell code is identical to the in-process vLLM cell — see
`registry.get_cell`, which lets `vllm_serve` reuse the `vllm_async` cell — so any map/perf difference is
attributable purely to the wire boundary.
"""
from __future__ import annotations

import time
import urllib.request

from .base import Backend


def _force_cpu_platform_when_no_gpu() -> None:
    """Make a GPU-less client build a CUDA-free meta VLLM model.

    vLLM only auto-selects its CPU platform for a `+cpu` build (or macOS); the stock GPU wheel on a
    GPU-less Linux host resolves to `UnspecifiedPlatform` and then fails to infer a device. So we
    register a CPU choice in vLLM's builtin platform registry that activates ONLY when NVML reports no
    GPU — leaving the GPU server (NVML count > 0) on the CUDA platform untouched. Must run before vLLM
    resolves `current_platform`; the serve backend calls it right before importing the vLLM model.
    """
    import vllm.platforms as platforms

    def cpu_when_no_gpu():
        try:
            from vllm.utils.import_utils import import_pynvml

            pynvml = import_pynvml()
            pynvml.nvmlInit()
            try:
                if pynvml.nvmlDeviceGetCount() > 0:
                    return None  # a GPU is present -> let the CUDA platform win (e.g. the server)
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass  # NVML absent/failed -> no usable GPU -> select the CPU platform
        return "vllm.platforms.cpu.CpuPlatform"

    platforms.builtin_platform_plugins["cpu"] = cpu_when_no_gpu


class VLLMServeBackend(Backend):
    name = "vllm_serve"

    HEALTH_TIMEOUT = 120.0  # seconds to wait for the server's /health before giving up

    def __init__(self, host: str, dtype: str | None = None):
        # `host` is the serve URL (e.g. "http://server:6677"). `dtype` is kept for interface parity
        # with the async backend's precision knob, but precision is the SERVER's engine config here
        # (the client is meta-only), so it is informational — not applied client-side.
        self.host = host.rstrip("/")
        self.dtype = dtype

    # No __getstate__ needed: unlike vllm_async (which carries a non-picklable event loop), this
    # backend holds only `host`/`dtype`. The cell's `build` closure captures `be.last` (a bound
    # method), so the backend IS serialized into the request sent to the server — and it must be, and
    # is, fully picklable.

    def load(self, repo: str):
        # Force vLLM's CPU platform when no GPU is visible BEFORE importing the vLLM model — otherwise
        # the meta build resolves the CUDA platform and crashes probing device capability (see project
        # memory). No-op on a GPU host. Must precede the vLLM import below.
        _force_cpu_platform_when_no_gpu()
        from nnsight.modeling.vllm import VLLM

        self._wait_for_health()
        # Meta model: architecture / envoy tree only — no engine, no weights, no GPU.
        return VLLM(repo)

    def _wait_for_health(self) -> None:
        url = f"{self.host}/health"
        deadline = time.monotonic() + self.HEALTH_TIMEOUT
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 (trusted compose host)
                    if r.status == 200:
                        return
            except Exception as e:  # 503 (engine warming) or connection-refused (not up yet) -> retry
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(
            f"nnsight-serve at {self.host} not healthy within {self.HEALTH_TIMEOUT}s: {last_err}"
        )

    def run(self, model, prompts, build):
        if isinstance(prompts, (list, tuple)) and len(prompts) > 1:
            # The server `asyncio.gather`s multiple invokes, but wiring N-invoke batched traces through
            # the serve cell is a follow-up — surface it honestly rather than silently dropping prompts.
            raise NotImplementedError(
                "batched (multi-prompt) serve is a follow-up; the serve cell submits one prompt/trace"
            )
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        with model.trace(prompt, serve=self.host, temperature=0.0, top_p=1, max_tokens=1):
            saved = build().save()  # var name "saved" IS the saves key; blocking push rebinds it here
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        # Two SEPARATE serve traces (two HTTP requests), mirroring the documented vLLM patching recipe
        # (no barrier shared across invokes). The clean activation comes back as a materialized CPU
        # tensor and is closed over by the corrupt trace's `patch(...)`, which serializes it into the
        # second request's intervention.
        with model.trace(clean_prompt, serve=self.host, temperature=0.0, top_p=1, max_tokens=1):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompt, serve=self.host, temperature=0.0, top_p=1, max_tokens=1):
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n=None):
        # The server runs vLLM in inference mode (activations are inference tensors with no autograd),
        # so attribution patching cannot run. Attempt `requires_grad_(True)` on the corrupt run's
        # activations: it raises on the worker and surfaces here as a clean per-cell ERROR. Forward-only
        # (no `.backward()` over the wire) so it fails fast with no hang risk — recording that the
        # `grad` primitive is unavailable over serve, same as in-process vLLM.
        with model.trace(corrupt_prompt, serve=self.host, temperature=0.0, top_p=1, max_tokens=1):
            acts = acts_of(model)
            for a in acts:
                a.requires_grad_(True)  # raises on the server: inference tensor, no autograd
            probe = acts[0].save()  # noqa: F841 — never meaningfully reached; the above raises
        return probe.detach().float().cpu()

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        # Whether a tracer.iter loop survives the serve transport (compiled client-side, iterated
        # server-side) is its own venue question — surface it honestly rather than guessing.
        raise NotImplementedError(
            "generation over serve is a follow-up; the per-step iteration loop is not yet wired "
            "through the serve transport"
        )

    def generate_patch(self, model, source_prompt, base_prompt, capture, build_step,
                       *, new_tokens, bounded=True):
        # Inherits the same gap as generate() over serve (the iteration loop is not wired through
        # the transport), plus the two-trace transplant — a follow-up for the serve venue.
        raise NotImplementedError(
            "generate_patch over serve is a follow-up; the per-step iteration loop is not yet "
            "wired through the serve transport"
        )

    def last(self, t):
        return t[-1:, :]  # server returns flat [tokens, vocab] (vLLM shape), like vllm_async

    def teardown(self, model) -> None:
        import gc

        try:
            del model
        finally:
            gc.collect()
        # No GPU cleanup: the client holds no engine. The server lifecycle is owned by docker-compose.

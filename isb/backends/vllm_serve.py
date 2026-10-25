"""vLLM serve-client backend: a meta `VLLM` model whose traces run on a separate nnsight-vllm-serve
instance over HTTP (`model.trace(serve=url)`). Saved values arrive materialized on the CPU and bind
back into the trace frame; worker errors surface as raised `RuntimeError`s. Several units are one
trace with one invoke per unit, as on the sync engine."""
from __future__ import annotations

import time
import urllib.request

from .vllm_base import VLLMBackend
from .vllm_sync import SAMPLING as _SAMPLING


def _force_cpu_platform_when_no_gpu() -> None:
    """Make a GPU-less client build a CUDA-free meta VLLM model: register a CPU platform choice in
    vLLM's builtin registry that activates only when NVML reports no GPU."""
    import vllm.platforms as platforms

    def cpu_when_no_gpu():
        try:
            from vllm.utils.import_utils import import_pynvml

            pynvml = import_pynvml()
            pynvml.nvmlInit()
            try:
                if pynvml.nvmlDeviceGetCount() > 0:
                    return None
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        return "vllm.platforms.cpu.CpuPlatform"

    platforms.builtin_platform_plugins["cpu"] = cpu_when_no_gpu


class VLLMServeBackend(VLLMBackend):
    name = "vllm_serve"

    HEALTH_TIMEOUT = 120.0  # seconds to wait for the server's /health

    def __init__(self, host: str, dtype: str | None = None, trust_remote_code: bool = False,
                 max_model_len: int | None = None, tokenizer: str | None = None,
                 enable_prefix_caching: bool | None = None):
        # dtype, max_model_len and enable_prefix_caching are the server's engine config, carried
        # for provenance; trust_remote_code applies to the meta build, which reads the repo config.
        super().__init__(dtype=dtype, trust_remote_code=trust_remote_code,
                         max_model_len=max_model_len, tokenizer=tokenizer,
                         enable_prefix_caching=enable_prefix_caching)
        self.host = host.rstrip("/")

    def load(self, repo: str):
        _force_cpu_platform_when_no_gpu()        # before vLLM resolves its platform
        from nnsight.modeling.vllm import VLLM

        self._wait_for_health()
        kw = {"trust_remote_code": True} if self.trust_remote_code else {}
        return VLLM(repo, **kw)                   # meta model: the envoy tree, no engine

    def _wait_for_health(self) -> None:
        url = f"{self.host}/health"
        deadline = time.monotonic() + self.HEALTH_TIMEOUT
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 (trusted compose host)
                    if r.status == 200:
                        return
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(
            f"nnsight-serve at {self.host} not healthy within {self.HEALTH_TIMEOUT}s: {last_err}")

    def _collect(self, rows, n, what):
        import torch

        mats = [r.detach().float().cpu() for r in rows if r is not None]
        if len(mats) != n:
            raise RuntimeError(f"vllm_serve {what} collected {len(mats)}/{n} per-unit outputs")
        return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)

    def run(self, model, prompts, build):
        prompts = self.many(prompts, "run")
        with model.trace(serve=self.host, **_SAMPLING) as tracer:
            rows = [None] * len(prompts)
            rows = rows.save()
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    rows[i] = build([i])
        return self._collect(rows, len(prompts), "run")

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        clean_prompts, corrupted_prompts = self.paired(clean_prompts, corrupted_prompts, "patch")
        n = len(clean_prompts)
        with model.trace(serve=self.host, **_SAMPLING) as tracer:
            clean = [None] * n
            clean = clean.save()
            for i, p in enumerate(clean_prompts):
                with tracer.invoke(p):
                    clean[i] = capture()
        clean_acts = [c.detach().float().cpu() for c in clean]
        with model.trace(serve=self.host, **_SAMPLING) as tracer:
            rows = [None] * n
            rows = rows.save()
            for i, p in enumerate(corrupted_prompts):
                with tracer.invoke(p):
                    rows[i] = patch(clean_acts[i])
        return self._collect(rows, n, "patch")

    def attribute(self, model, clean_prompts, corrupt_prompts, acts_of, metric_of, n):
        """The server runs vLLM in inference mode; `requires_grad_` on an activation raises there
        and surfaces here, forward-only, as the recorded verdict."""
        clean_prompts, corrupt_prompts = self.paired(clean_prompts, corrupt_prompts, "attribute")
        probes = [None] * len(corrupt_prompts)
        with model.trace(serve=self.host, **_SAMPLING) as tracer:
            probes = probes.save()
            for i, p in enumerate(corrupt_prompts):
                with tracer.invoke(p):
                    acts = acts_of(model)
                    for a in acts:
                        a.requires_grad_(True)
                    probes[i] = acts[0]       # not reached: the line above raises
        return self._collect(probes, len(corrupt_prompts), "attribute")

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        raise NotImplementedError("the per-step iteration loop is not wired through the serve transport")

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        raise NotImplementedError("the per-step iteration loop is not wired through the serve transport")

    def plain(self, model, units, *, new_tokens=1):
        raise NotImplementedError("the serve client holds no engine for a plain request")

    def last(self, t):
        return t[-1:, :]  # the server returns vLLM's flat [tokens, vocab]

    def teardown(self, model) -> None:
        import gc

        try:
            del model
        finally:
            gc.collect()

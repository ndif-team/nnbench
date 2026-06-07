"""vLLM async backend infra (design.md F2; §12.3).

`run` owns the `with model.trace(...)` and the async stream in one coroutine; the cell's
`build()` closure runs inside the trace. After the with-exit submits the request, we
stream to the finished output and pull the saved value from its nested `.saves`,
surfacing nnsight's deferred-exception payload as a clean error.

NOTE: multi-prompt under vLLM (continuous batching) is a future step — this handles one
prompt; the cell interface already accepts a list.
"""
from __future__ import annotations

import contextlib

from .base import Backend


class VLLMAsyncBackend(Backend):
    name = "vllm_async"

    def __init__(self, dtype: str | None = None):
        # dtype is a real engine-config axis (design L3). Default None -> vLLM's own default
        # (bf16 for GPT-2). Forcing "float32" matches HF's precision, which is how we separate a
        # precision-degradation (SUPPORTED_DEGRADED) from a true mechanism bug (SILENTLY_WRONG).
        self.dtype = dtype

    def load(self, repo: str, gpu_memory_utilization: float = 0.2):
        from nnsight.modeling.vllm import VLLM

        kw = {} if self.dtype is None else {"dtype": self.dtype}
        return VLLM(
            repo, mode="async", dispatch=True,
            gpu_memory_utilization=gpu_memory_utilization, **kw,
        )

    def run(self, model, prompts, build):
        import asyncio

        if isinstance(prompts, (list, tuple)) and len(prompts) > 1:
            # fail loud rather than silently scoring 1 prompt vs HF's N (-> false mislabel)
            raise NotImplementedError(
                "vllm_async multi-prompt (continuous batching) not wired yet"
            )
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts

        async def _go():
            with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=1) as tracer:
                proxy = build()
                saved = proxy.save()  # noqa: F841  — the var name IS the async .saves key
            last = None
            async for output in tracer.backend:   # with-exit already submitted
                last = output
            return self._extract(last)

        return asyncio.run(_go())

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        import asyncio

        async def _go():
            # both traces in ONE event loop / coroutine: two sequential single-prompt requests on
            # the same async engine. Avoids the two-`asyncio.run` two-loop hazard, and uses two
            # separate traces (not two invokes) because vLLM does not share a barrier / cross-invoke
            # value across invokes — the documented patching recipe (VLLM_GUIDE "Activation Patching").
            with model.trace(clean_prompt, temperature=0.0, top_p=1, max_tokens=1) as t1:
                ca = capture().save()  # noqa: F841 — var name IS the async .saves key
            last1 = None
            async for output in t1.backend:
                last1 = output
            clean_act = self._extract(last1)        # materialized CPU tensor

            with model.trace(corrupted_prompt, temperature=0.0, top_p=1, max_tokens=1) as t2:
                res = patch(clean_act).save()  # noqa: F841
            last2 = None
            async for output in t2.backend:
                last2 = output
            return self._extract(last2)

        return asyncio.run(_go())

    def last(self, t):
        return t[-1:, :]                          # flat [tokens, vocab] -> [1, vocab]

    def _extract(self, out):
        saves = out.saves
        while (
            isinstance(saves, dict)
            and len(saves) == 1
            and isinstance(next(iter(saves.values())), dict)
        ):
            saves = next(iter(saves.values()))
        if isinstance(saves, dict) and {"type_name", "message", "traceback"} <= set(saves):
            msg = (saves.get("message") or "").strip().splitlines()
            raise RuntimeError(
                f"worker intervention: {msg[-1] if msg else saves['type_name']}"
            )
        v = saves["saved"] if "saved" in saves else next(iter(saves.values()))
        return v.detach().float().cpu()

    def teardown(self, model) -> None:
        import gc

        import torch

        with contextlib.suppress(Exception):
            model.vllm_entrypoint.shutdown()
        with contextlib.suppress(Exception):
            del model
        gc.collect()
        with contextlib.suppress(Exception):
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

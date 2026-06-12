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
        self._loop = None

    def __getstate__(self):
        # The build closure captures `self.last` (a bound method), so this backend is serialized into
        # the mediator sent to the worker. The persistent event loop is not picklable (it holds a
        # `_contextvars.Context`) and the worker never needs it — drop it from the pickle.
        state = self.__dict__.copy()
        state["_loop"] = None
        return state

    def _run_coro(self, coro):
        """Drive an async trace on ONE persistent event loop, reused across calls. `asyncio.run`
        creates AND closes a fresh loop each call, which kills the AsyncLLM engine's background loop
        and makes the next call raise EngineDeadError — fatal once the engine is amortized across
        warmup + timed trials (the perf path). So we keep one loop alive for the backend's lifetime.

        We still run each call inside a FRESH copied contextvars Context — exactly what `asyncio.run`
        does (`copy_context().run(...)`). Without this, `run_until_complete` executes in the live
        current context and nnsight's mediator serialization captures a `_contextvars.Context`
        (`TypeError: cannot pickle '_contextvars.Context'`). The copy isolates each call's context."""
        import asyncio
        import contextvars

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return contextvars.copy_context().run(self._loop.run_until_complete, coro)

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
            return self.run_batched(model, prompts, build)        # N prompts -> continuous batching
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts

        async def _go():
            with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=1) as tracer:
                proxy = build()
                saved = proxy.save()  # noqa: F841  — the var name IS the async .saves key
            last = None
            async for output in tracer.backend:   # with-exit already submitted
                last = output
            return self._extract(last)

        return self._run_coro(_go())

    def run_batched(self, model, prompts, build):
        """Batched throughput path: N prompts in ONE async trace via per-prompt `tracer.invoke`
        (the documented multi-invoke pattern). Each invoke runs `build()` on its own prompt's
        activations and stores it in a shared parent-scope list; after draining, the per-prompt
        outputs are concatenated along the batch dim so they match HF's `[..., N, vocab]` and the
        oracle can compare per prompt (throughput = N / batch-latency).

        Depends on the upstream async multi-prompt submission. Until that lands the engine submits
        only the first invoke, so this returns fewer than N per-prompt outputs and the cell is
        flagged by the oracle — surfaced, not silently passed.
        """
        import asyncio

        import torch

        n = len(prompts)

        async def _go():
            with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
                rows = [None] * n
                rows = rows.save()  # noqa: F841 — parent-scope list; "rows" is the saves key
                for i, p in enumerate(prompts):
                    with tracer.invoke(p):
                        rows[i] = build()
            last = None
            async for output in tracer.backend:
                last = output
            collected = self._extract_list(last)                  # ordered, len n (None where not produced)
            mats = [r.detach().float().cpu() for r in collected if r is not None]
            if not mats:
                raise RuntimeError(
                    "vllm_async batched produced no per-prompt outputs "
                    "(awaiting the upstream async multi-prompt submission fix)"
                )
            # each per-prompt output is [..., 1, vocab]; concat on the batch dim -> [..., N, vocab]
            return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)

        return self._run_coro(_go())

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

        return self._run_coro(_go())

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n=None):
        """vLLM activations are inference-mode tensors with no autograd, so attribution patching
        cannot run. We attempt `requires_grad_(True)` on the corrupt run's residuals: it raises in the
        worker (`Setting requires_grad=True on inference tensor ...`) and surfaces as a clean per-cell
        ERROR. Deliberately FORWARD-ONLY — no `.backward()` over the async path — so it fails fast
        with no hang risk; the point is to record that the `grad` primitive is unavailable on vLLM."""
        async def _go():
            with model.trace(corrupt_prompt, temperature=0.0, top_p=1, max_tokens=1) as tracer:
                acts = acts_of(model)
                for a in acts:
                    a.requires_grad_(True)            # raises: inference tensor, no autograd
                probe = acts[0].save()  # noqa: F841 — never meaningfully reached; the above raises
            last = None
            async for output in tracer.backend:
                last = output
            return self._extract(last)                # surfaces the worker's requires_grad error
        return self._run_coro(_go())

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        import torch

        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts

        async def _go():
            with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=new_tokens) as tracer:
                rows = list().save()  # noqa: F841 — "rows" is the async .saves key; create the
                # list in THIS frame and let build_step mutate it: the saves collector reads
                # trace-frame locals, a save made in a nested frame is silently lost
                steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
                for _step in steps:
                    rows.append(build_step())
            last = None
            async for output in tracer.backend:
                last = output
            if not hasattr(last, "saves"):
                # the unbounded realization: the vLLM path never sets a stop bound, the loop
                # overruns and is unwound by Cancelation BEFORE the body's final push — the
                # finished output carries no saves (F-13; nnsight vllm-construct-gaps §1)
                raise RuntimeError(
                    "vllm_async generation: finished output carried no saves — unbounded "
                    "tracer.iter[:] drops all per-step saves on the vLLM path (F-13)"
                )
            saves = last.saves
            while (
                isinstance(saves, dict)
                and len(saves) == 1
                and isinstance(next(iter(saves.values())), dict)
                and not {"type_name", "message", "traceback"} <= set(saves)
            ):
                saves = next(iter(saves.values()))
            if isinstance(saves, dict) and {"type_name", "message", "traceback"} <= set(saves):
                msg = (saves.get("message") or "").strip().splitlines()
                raise RuntimeError(
                    f"worker intervention: {msg[-1] if msg else saves['type_name']}"
                )
            v = saves["rows"] if isinstance(saves, dict) and "rows" in saves else (
                next(iter(saves.values())) if isinstance(saves, dict) else saves
            )
            mats = [r.detach().float().cpu() for r in v if r is not None]
            if len(mats) != new_tokens:
                raise RuntimeError(
                    f"vllm_async generation collected {len(mats)} per-step rows, "
                    f"expected {new_tokens}")
            return torch.cat(mats, dim=0)             # per-step [1, vocab] -> [steps, vocab]

        return self._run_coro(_go())

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

    def _extract_list(self, out):
        """Like `_extract` but the saved value is the batched per-prompt list (`rows`); return it as
        a Python list (one entry per invoke; entries may be None if not produced)."""
        if not hasattr(out, "saves"):
            # nnsight attaches `.saves` only on a finished output whose saves were collected. A
            # multi-invoke batched trace currently submits fewer requests than invokes upstream, so
            # the list is never collected — surface that clearly rather than as a raw AttributeError.
            raise RuntimeError(
                "vllm_async batched: finished output carried no saves — the engine submitted fewer "
                "requests than invokes (awaiting the upstream async multi-prompt submission fix)"
            )
        saves = out.saves
        while (
            isinstance(saves, dict)
            and len(saves) == 1
            and isinstance(next(iter(saves.values())), dict)
            and not {"type_name", "message", "traceback"} <= set(saves)
        ):
            saves = next(iter(saves.values()))
        if isinstance(saves, dict) and {"type_name", "message", "traceback"} <= set(saves):
            msg = (saves.get("message") or "").strip().splitlines()
            raise RuntimeError(
                f"worker intervention: {msg[-1] if msg else saves['type_name']}"
            )
        v = saves.get("rows") if isinstance(saves, dict) and "rows" in saves else (
            next(iter(saves.values())) if isinstance(saves, dict) else saves
        )
        return list(v) if isinstance(v, (list, tuple)) else [v]

    def teardown(self, model) -> None:
        import gc

        import torch

        with contextlib.suppress(Exception):
            model.vllm_entrypoint.shutdown()
        if self._loop is not None and not self._loop.is_closed():   # close the loop AFTER engine shutdown
            with contextlib.suppress(Exception):
                self._loop.close()
            self._loop = None
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

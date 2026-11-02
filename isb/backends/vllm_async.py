"""vLLM async backend: nnsight's `VLLM(mode="async")`, the in-process AsyncLLM engine.

The async path submits one engine request per trace, so several units run as several single-unit
coroutines gathered on one persistent event loop; the engine batches them. Each coroutine owns the
`with model.trace(...)`, drains `tracer.backend` to the finished output and reads the saved value
from its nested `.saves`.
"""
from __future__ import annotations

import contextlib

from .base import prompts_of
from .vllm_base import VLLMBackend

_SAMPLING = {"temperature": 0.0, "top_p": 1, "max_tokens": 1}


class VLLMAsyncBackend(VLLMBackend):
    name = "vllm_async"

    def __init__(self, dtype: str | None = None,
                 trust_remote_code: bool = False,
                 pipeline_parallel_size: int = 1,
                 tensor_parallel_size: int = 1,
                 distributed_executor_backend: str | None = None,
                 gpu_memory_utilization: float = 0.2,
                 max_model_len: int | None = None,
                 tokenizer: str | None = None,
                 enable_prefix_caching: bool | None = None):
        super().__init__(dtype=dtype, trust_remote_code=trust_remote_code,
                         max_model_len=max_model_len, tokenizer=tokenizer,
                         enable_prefix_caching=enable_prefix_caching)
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self._loop = None
        self._loop_thread = None         # set while a concurrent session runs the loop

    def __getstate__(self):
        # A build closure captures `self.last`, so this backend travels inside the mediator; the
        # event loop and its thread are not picklable and the worker never needs them.
        state = self.__dict__.copy()
        state["_loop"] = None
        state["_loop_thread"] = None
        return state

    def _run_coro(self, coro):
        """Drive a coroutine on the one persistent loop (a fresh loop per call would close the
        engine's background loop), in a copied contextvars context as `asyncio.run` does."""
        import asyncio
        import contextvars

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        if self._loop_thread is not None:            # a concurrent session: the loop runs there
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return contextvars.copy_context().run(self._loop.run_until_complete, coro)

    def _gather(self, coros):
        """Run the coroutines together and return their results in order."""
        import asyncio

        async def _all():
            return await asyncio.gather(*coros)

        return self._run_coro(_all())

    # Several client threads may submit at once: AsyncLLM is a serving engine.
    concurrent_requests = True

    @contextlib.contextmanager
    def concurrent_session(self):
        """Run the persistent loop on its own thread while clients submit from theirs."""
        import asyncio
        import threading

        if self._loop_thread is not None:
            raise RuntimeError("a concurrent session is already open")
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        loop = self._loop
        thread = threading.Thread(target=loop.run_forever, name="isb-vllm-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        try:
            yield
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._loop_thread = None

    def load(self, repo: str, gpu_memory_utilization: float | None = None):
        from nnsight.modeling.vllm import VLLM

        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
        kw = self._engine_kwargs()
        if self.pipeline_parallel_size > 1:
            kw["pipeline_parallel_size"] = self.pipeline_parallel_size
        if self.tensor_parallel_size > 1:
            kw["tensor_parallel_size"] = self.tensor_parallel_size
        if self.distributed_executor_backend is not None:
            kw["distributed_executor_backend"] = self.distributed_executor_backend
        return VLLM(repo, mode="async", dispatch=True,
                    gpu_memory_utilization=gpu_memory_utilization, **kw)

    @staticmethod
    async def _finished(tracer):
        last = None
        async for output in tracer.backend:          # the with-exit already submitted the request
            last = output
        return last

    def _cat(self, mats):
        import torch

        return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)

    def run(self, model, prompts, build, *, keep_dtype=False):
        prompts = self.many(prompts, "run")

        async def one(i, prompt):
            with model.trace(prompt, **_SAMPLING) as tracer:
                saved = build([i]).save()  # noqa: F841  (the variable name is the saves key)
            return self._extract(await self._finished(tracer), keep_dtype)

        return self._cat(self._gather([one(i, p) for i, p in enumerate(prompts)]))

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        pairs = list(zip(*self.paired(clean_prompts, corrupted_prompts, "patch")))

        async def one(clean_prompt, corrupted_prompt):
            with model.trace(clean_prompt, **_SAMPLING) as t1:
                ca = capture().save()  # noqa: F841
            clean_act = self._extract(await self._finished(t1))
            with model.trace(corrupted_prompt, **_SAMPLING) as t2:
                res = patch(clean_act).save()  # noqa: F841
            return self._extract(await self._finished(t2))

        return self._cat(self._gather([one(c, k) for c, k in pairs]))

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        """One pair per pair of traces; the pairs' gradients accumulate on `step`'s trainables."""
        import torch

        pairs = list(zip(*self.paired(source_prompts, base_prompts, "train_patch")))

        async def _go():
            total = torch.zeros(())
            for i, (source_prompt, base_prompt) in enumerate(pairs):
                with model.trace(source_prompt, **_SAMPLING) as t1:
                    s = capture().save()  # noqa: F841
                src = self._extract(await self._finished(t1))
                with model.trace(base_prompt, **_SAMPLING) as t2:
                    loss = step(src, [i])
                    lv = loss.save()  # noqa: F841
                    with loss.backward():
                        pass
                total = total + self._extract(await self._finished(t2)).reshape(())
            return total

        return self._run_coro(_go())

    def vjp_batch(self, model, prompts, acts_of, target_of, make_cotangent, n):
        import torch

        prompts = self.many(prompts, "vjp_batch")

        async def one(i, prompt):
            with model.trace(prompt, **_SAMPLING) as tracer:
                grads = [None] * n
                grads = grads.save()  # noqa: F841
                acts = acts_of(model)
                for a in acts:
                    a.requires_grad_(True)
                target = target_of(model)
                cot = make_cotangent(target, [i])
                metric = (target * cot).sum()
                with metric.backward():
                    for L in range(len(acts) - 1, -1, -1):
                        grads[L] = acts[L].grad
            (out,) = self._extract_keys(await self._finished(tracer), ["grads"])
            return [g.detach().float().cpu() for g in out]

        per_prompt = self._gather([one(i, p) for i, p in enumerate(prompts)])
        return [torch.stack([p[L] for p in per_prompt]) for L in range(n)]      # n x [B, seq, d]

    def attribute(self, model, clean_prompts, corrupt_prompts, acts_of, metric_of, n):
        """The same flow as HF's; the engine's own first grad refusal is the recorded verdict."""
        import torch

        pairs = list(zip(*self.paired(clean_prompts, corrupt_prompts, "attribute")))

        async def one(i, clean_prompt, corrupt_prompt):
            with model.trace(clean_prompt, **_SAMPLING) as t1:
                clean = [None] * n
                clean = clean.save()  # noqa: F841
                a = acts_of(model)
                for L in range(n):
                    clean[L] = a[L]
            (clean_acts,) = self._extract_keys(await self._finished(t1), ["clean"])
            with model.trace(corrupt_prompt, **_SAMPLING) as t2:
                corrupt = [None] * n
                corrupt = corrupt.save()  # noqa: F841
                grads = [None] * n
                grads = grads.save()  # noqa: F841
                a = acts_of(model)
                for L in range(n):
                    a[L].requires_grad_(True)
                for L in range(n):
                    corrupt[L] = a[L]
                metric = metric_of(model, i)
                with metric.sum().backward():
                    for L in range(n - 1, -1, -1):
                        grads[L] = a[L].grad
            corrupt_acts, grad_acts = self._extract_keys(await self._finished(t2), ["corrupt", "grads"])
            return torch.stack([
                ((clean_acts[L].float() - corrupt_acts[L].float()) * grad_acts[L].float())
                .sum().detach().cpu() for L in range(n)])

        return torch.stack(self._gather([one(i, c, k) for i, (c, k) in enumerate(pairs)]))

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        import torch

        prompts = self.many(prompts, "generate")

        async def one(prompt):
            with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=new_tokens,
                             min_tokens=new_tokens) as tracer:
                rows = list().save()  # noqa: F841  (made in the trace frame; appended per step)
                steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
                for step in steps:
                    rows.append(build_step(step))
            return self._extract_step_rows(await self._finished(tracer), new_tokens, "generation")

        return torch.cat(self._gather([one(p) for p in prompts]), dim=1)

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        import torch

        pairs = list(zip(*self.paired(source_prompts, base_prompts, "generate_patch")))

        async def one(source_prompt, base_prompt):
            with model.trace(source_prompt, **_SAMPLING) as t1:
                ca = capture().save()  # noqa: F841
            clean_act = self._extract(await self._finished(t1))
            with model.trace(base_prompt, temperature=0.0, top_p=1, max_tokens=new_tokens,
                             min_tokens=new_tokens) as tracer:
                rows = list().save()  # noqa: F841
                steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
                for step in steps:
                    rows.append(build_step(clean_act, step))
            return self._extract_step_rows(await self._finished(tracer), new_tokens, "generate_patch")

        return torch.cat(self._gather([one(s, b) for s, b in pairs]), dim=1)

    def plain(self, model, units, *, new_tokens=1):
        """One request per prompt straight to nnsight's AsyncLLM entrypoint, submitted together."""
        import uuid

        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, top_p=1, max_tokens=new_tokens,
                                min_tokens=new_tokens)

        async def one(prompt):
            async for _ in model.vllm_entrypoint.generate(prompt, params, uuid.uuid4().hex):
                pass

        self._gather([one(p) for p in prompts_of(units)])

    def last(self, t):
        return t[-1:, :]                          # flat [tokens, vocab] -> [1, vocab]

    def collective_rpc(self, model, method, args=()):
        """Through `model.vllm_entrypoint`, the `AsyncLLM` nnsight runs (docs/models/vllm.md), on
        the backend's persistent loop."""
        return self._run_coro(model.vllm_entrypoint.collective_rpc(method, args=args))

    def _saves(self, out, what):
        """The finished output's flat `{name: value}` saves. nnsight 0.8 raises a worker error
        itself; nnsight 0.7 returns it in the saves as `__nnsight_exceptions__: {request_id:
        {type_name, message, traceback, is_control_flow}}`, raised here as nnsight's own client
        does, skipping control-flow entries such as `tracer.stop()`."""
        if not hasattr(out, "saves"):
            raise RuntimeError(f"vllm_async {what}: the finished output carried no saves")
        saves = dict(out.saves)
        for error in saves.pop("__nnsight_exceptions__", {}).values():
            if not error.get("is_control_flow", False):
                lines = (error.get("message") or "").strip().splitlines()
                raise RuntimeError(f"worker intervention: {lines[-1] if lines else error['type_name']}")
        return saves

    def _extract_keys(self, out, keys):
        saves = self._saves(out, "trace")
        return [saves[k] for k in keys]

    def _extract(self, out, keep_dtype=False):
        saves = self._saves(out, "trace")
        v = saves["saved"] if "saved" in saves else next(iter(saves.values()))
        return (v.detach() if keep_dtype else v.detach().float()).cpu()

    def _extract_step_rows(self, out, new_tokens, what):
        """The per-step `rows` of one request -> `[new_tokens, 1, ...]`."""
        import torch

        saves = self._saves(out, what)
        v = saves["rows"] if isinstance(saves, dict) and "rows" in saves else (
            next(iter(saves.values())) if isinstance(saves, dict) else saves)
        mats = [r.detach().float().cpu() for r in v if r is not None]
        if len(mats) != new_tokens:
            raise RuntimeError(
                f"vllm_async {what} collected {len(mats)} per-step rows, expected {new_tokens}")
        return torch.stack(mats, dim=0)

    def teardown(self, model) -> None:
        import asyncio
        import gc

        import torch

        with contextlib.suppress(Exception):
            model.vllm_entrypoint.shutdown()
        if self._loop is not None and not self._loop.is_closed():
            # let the engine handlers' finally blocks run before the loop closes
            async def drain():
                pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                await self._loop.shutdown_asyncgens()

            with contextlib.suppress(Exception):
                self._loop.run_until_complete(drain())
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

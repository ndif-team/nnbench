"""vLLM async backend (design.md F2 = HF + vLLM-async).

Async save-collection differs from HF: saves come back on the finished RequestOutput's
`.saves` dict (keyed by the inferred var name of the `.save()` assignment), not via the
handle. We name the assignment `saved` so the key is stable; we also fall back to the
sole save value if there is exactly one.
"""
from __future__ import annotations

import contextlib
import time

from .base import Backend, BackendCtx


class VLLMAsyncBackend(Backend):
    name = "vllm_async"

    def load(self, repo: str, gpu_memory_utilization: float = 0.2):
        from nnsight.modeling.vllm import VLLM

        # enforce_eager is forced True internally (CUDA graphs ⊥ hooks); device via
        # CUDA_VISIBLE_DEVICES set by the caller.
        return VLLM(
            repo,
            mode="async",
            dispatch=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def run(self, model, program, prompt: str, generation) -> dict:
        import asyncio

        import torch

        ctx = BackendCtx(
            select_last=lambda t: t[-1:, :],              # flat [tokens, H] -> [1, vocab]
            stack=lambda rows: torch.stack(rows, dim=0),  # -> [n_sites, 1, vocab]
        )
        gen_tokens = max(1, generation.new_tokens)

        async def _go():
            with model.trace(
                prompt, temperature=0.0, top_p=1, max_tokens=gen_tokens
            ) as tracer:
                proxy = program.build_proxy(model, ctx)
                saved = proxy.save()  # noqa: F841  (named for the async .saves key)
            # with-exit auto-submits; stream to the finished output (saves attached there)
            last = None
            async for output in tracer.backend:
                last = output
            return last

        t0 = time.time()
        out = asyncio.run(_go())
        latency = time.time() - t0
        return {
            "value": self._collect(out),
            "site_ids": program.site_ids,
            "latency_s": latency,
        }

    def _collect(self, out):
        # async returns {base_id: {var_name: value}} (per-request namespacing); descend
        # the single-key wrapper layers until we reach the var->value dict.
        saves = out.saves
        while (
            isinstance(saves, dict)
            and len(saves) == 1
            and isinstance(next(iter(saves.values())), dict)
        ):
            saves = next(iter(saves.values()))
        # nnsight's deferred-exception payload: an intervention raised in the worker.
        if isinstance(saves, dict) and {"type_name", "message", "traceback"} <= set(saves):
            import sys

            print("--- vLLM worker intervention traceback ---", file=sys.stderr)
            print(saves.get("traceback", ""), file=sys.stderr)
            # the message embeds the inner traceback; surface just its final line
            msg = (saves.get("message") or "").strip().splitlines()
            reason = msg[-1] if msg else saves["type_name"]
            raise RuntimeError(f"worker intervention: {reason}")
        if "saved" in saves:
            v = saves["saved"]
        elif len(saves) == 1:
            v = next(iter(saves.values()))
        else:
            raise RuntimeError(f"unexpected async save keys: {list(saves)}")
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

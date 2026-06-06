"""HF Transformers backend — also the equivalence oracle's reference (design.md §8.3)."""
from __future__ import annotations

import time

from .base import Backend, BackendCtx


class HFBackend(Backend):
    name = "hf"

    def load(self, repo: str, device: str = "cuda:0"):
        from nnsight import LanguageModel

        # eager attention so this stays a faithful "HF-eager reference" (§11.8) — matters
        # once tier-(c) attn.weights workloads run; harmless for logit lens.
        return LanguageModel(
            repo, device_map=device, dispatch=True, attn_implementation="eager"
        )

    def run(self, model, program, prompt: str, generation) -> dict:
        import torch

        ctx = BackendCtx(
            select_last=lambda t: t[:, -1, :],            # [B, S, H] -> [B, vocab]
            stack=lambda rows: torch.stack(rows, dim=0),  # -> [n_sites, B, vocab]
        )
        t0 = time.time()
        with model.trace(prompt):
            proxy = program.build_proxy(model, ctx)
            saved = proxy.save()
        latency = time.time() - t0
        return {
            "value": saved.detach().float().cpu(),
            "site_ids": program.site_ids,
            "latency_s": latency,
        }

    def teardown(self, model) -> None:
        import gc

        import torch

        try:
            del model
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

"""HF Transformers backend infra — also the per-family oracle control (design.md §12.2)."""
from __future__ import annotations

from .base import Backend


class HFBackend(Backend):
    name = "hf"

    def load(self, repo: str, device: str = "cuda:0"):
        from nnsight import LanguageModel

        return LanguageModel(
            repo, device_map=device, dispatch=True, attn_implementation="eager"
        )

    def run(self, model, prompts, build):
        with model.trace(prompts):          # a list of prompts -> a batch
            saved = build().save()
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        with model.trace(clean_prompt):     # trace 1: snapshot the clean activation
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompt):  # trace 2: inject it, observe the corrupted run
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def last(self, t):
        return t[:, -1, :]                  # [B, S, vocab] -> [B, vocab]

    def teardown(self, model) -> None:
        import gc

        import torch

        try:
            del model
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

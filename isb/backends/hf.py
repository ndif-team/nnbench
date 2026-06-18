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

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n):
        import torch

        # Pre-create the result lists OUTSIDE the trace and index-assign inside: the trace body is
        # compiled into a separate function, so rebinding a name inside it (`clean = [...]`) is local
        # to that body and lost, but index-assigning into a list made out here mutates the shared
        # object — the pattern nnsight's attribution-patching doc uses.
        clean = [None] * n
        with model.trace(clean_prompt):                  # pass 1: clean activations
            a = acts_of(model)
            for L in range(n):
                clean[L] = a[L].save()

        corrupt = [None] * n
        grads = [None] * n
        with model.trace(corrupt_prompt):                # pass 2: corrupt forward + backward
            a = acts_of(model)
            for L in range(n):
                a[L].requires_grad_(True)                # retain grad on the intermediate residuals
            for L in range(n):
                corrupt[L] = a[L].save()
            metric = metric_of(model)                    # read .output BEFORE the backward session
            with metric.sum().backward():
                for L in range(n - 1, -1, -1):           # grads in REVERSE module order
                    grads[L] = a[L].grad.save()

        # attribution[L] = (clean - corrupt) · grad, summed over the activation's dims
        return torch.stack([
            ((clean[L].float() - corrupt[L].float()) * grads[L].float()).sum().detach().cpu()
            for L in range(n)
        ])

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        import torch

        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        it = slice(0, new_tokens) if bounded else slice(None)   # unbounded: stop bound comes from
        with model.generate(prompt, max_new_tokens=new_tokens,  # max_new_tokens via default_all
                            do_sample=False) as tracer:
            rows = list().save()
            for _step in tracer.iter[it]:
                rows.append(build_step())
        if len(rows) != new_tokens:
            raise RuntimeError(
                f"hf generation collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)   # [steps, vocab]

    def generate_patch(self, model, source_prompt, base_prompt, capture, build_step,
                       *, new_tokens, bounded=True):
        import torch

        with model.trace(source_prompt):            # trace 1: snapshot the source activation
            ca = capture().save()
        clean_act = ca.detach().float().cpu()       # materialized; closed over by trace 2's steps

        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(base_prompt, max_new_tokens=new_tokens, do_sample=False) as tracer:
            rows = list().save()
            for _step in tracer.iter[it]:
                rows.append(build_step(clean_act))  # inject@prefill + read this step's logits
        if len(rows) != new_tokens:
            raise RuntimeError(
                f"hf generate_patch collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)   # [steps, vocab]

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

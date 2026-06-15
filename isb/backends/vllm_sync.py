"""vLLM **sync** backend infra (design.md §12.3) — the in-process synchronous engine.

Where `vllm_async` submits a request and drains `tracer.backend` for the finished output's
nested `.saves`, the SYNC engine runs the trace in-process and pushes saved values back into
the trace frame — exactly HF's access pattern. So this backend mirrors `hf.py` (read
`saved.detach()` after the `with`-block), differing only in: `VLLM(mode="sync")` load,
sampling kwargs on `trace(...)`, vLLM logits shape (`last`), and vLLM engine teardown.

`vllm_sync` is the sync engine mode of the vLLM backend (the engine-mode axis of the L3 sweep,
design §6), alongside `vllm_async`.
"""
from __future__ import annotations

import contextlib

from .base import Backend

# Greedy, single forward step — the deterministic regime the oracle compares against.
_SAMPLING = {"temperature": 0.0, "top_p": 1, "max_tokens": 1}


class VLLMSyncBackend(Backend):
    name = "vllm_sync"

    def __init__(self, dtype: str | None = None):
        # dtype is a real engine-config axis (design L3); None -> vLLM's default (bf16 for
        # GPT-2). Forcing "float32" matches HF precision, separating DEGRADED from SILENTLY_WRONG.
        self.dtype = dtype

    def load(self, repo: str, gpu_memory_utilization: float = 0.2):
        from nnsight.modeling.vllm import VLLM

        kw = {} if self.dtype is None else {"dtype": self.dtype}
        return VLLM(
            repo, mode="sync", dispatch=True,
            gpu_memory_utilization=gpu_memory_utilization, **kw,
        )

    def run(self, model, prompts, build):
        """Single synchronous trace; the saved value binds in-frame (the sync engine
        push_variables the worker's saves back into the trace frame). N prompts -> the
        multi-invoke batched path (works on sync; the async submission gate is async-only)."""
        if isinstance(prompts, (list, tuple)) and len(prompts) > 1:
            return self.run_batched(model, prompts, build)
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        with model.trace(prompt, **_SAMPLING):
            saved = build().save()
        return saved.detach().float().cpu()

    def run_batched(self, model, prompts, build):
        """Batched throughput path: N prompts in ONE sync trace via per-prompt `tracer.invoke`
        (the documented multi-invoke pattern). Each invoke runs `build()` on its own prompt and
        stores it in a shared parent-scope list; the per-prompt outputs are concatenated along the
        batch dim to match HF's `[..., N, vocab]` so the oracle compares per prompt.

        This is the cross-invoke construct the barrier fix unblocked on the sync engine — but
        SIMPLER than barrier (independent invokes, no cross-invoke value hand-off). vLLM runs each
        invoke as its OWN request (continuous batching, no left-padding), so unlike HF's padded
        batch it carries no absolute-position artifact. The async engine cannot do this (the
        multi-prompt submission gate); the sync engine submits all invokes."""
        import torch

        n = len(prompts)
        with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            rows = [None] * n
            rows = rows.save()  # noqa: F841 — parent-scope list; the save collector reads it in-frame
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    rows[i] = build()
        mats = [r.detach().float().cpu() for r in rows if r is not None]
        if len(mats) != n:
            raise RuntimeError(
                f"vllm_sync batched collected {len(mats)}/{n} per-prompt outputs "
                "(a multi-invoke submission or save-collection gap)")
        # each per-prompt output is [..., 1, vocab]; concat on the batch dim -> [..., N, vocab]
        return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        """Two-trace activation patching: capture from a clean single-prompt trace, inject in a
        corrupted one. Two sequential sync traces (like HF), so no multi-invoke/barrier needed."""
        with model.trace(clean_prompt, **_SAMPLING):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompt, **_SAMPLING):
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n=None):
        """sync vLLM activations are inference-mode tensors with no autograd (same as async), so
        attribution patching cannot run: `requires_grad_(True)` raises in the worker and surfaces
        as a clean per-cell ERROR. Deliberately FORWARD-ONLY — no `.backward()` — so it fails fast
        with no hang risk; the point is to record that the `grad` primitive is unavailable on vLLM."""
        with model.trace(corrupt_prompt, **_SAMPLING):
            acts = acts_of(model)
            for a in acts:
                a.requires_grad_(True)            # raises: inference tensor, no autograd
            probe = acts[0].save()  # noqa: F841 — never meaningfully reached; the above raises
        return probe.detach().float().cpu()       # surfaces the worker's requires_grad error

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        """Per-step decode reads/intervention. With the construct-gap fix, the UNBOUNDED form
        (`tracer.iter[:]`) preserves its saves on sync (it dropped them all pre-fix), so both
        realizations collect `new_tokens` rows here."""
        import torch

        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=new_tokens) as tracer:
            rows = list().save()
            steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
            for _step in steps:
                rows.append(build_step())
        if len(rows) != new_tokens:
            raise RuntimeError(
                f"vllm_sync generation collected {len(rows)} per-step rows, "
                f"expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)   # [steps, vocab]

    def last(self, t):
        return t[-1:, :]                          # flat [tokens, vocab] -> [1, vocab] (vLLM shape)

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

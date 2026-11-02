"""vLLM sync backend: nnsight's `VLLM(mode="sync")`, the in-process synchronous engine.

Several units are one trace with one `tracer.invoke(prompt)` per unit (one request each; the
engine batches them); a container saved above the invokes gathers them by pre-sized slot. Saved
values bind back into the trace frame, so this mirrors `hf.py`, differing in the per-invoke form,
the sampling kwargs and vLLM's flat `[tokens, ...]` shapes.
"""
from __future__ import annotations

import contextlib

from .base import prompts_of
from .vllm_base import VLLMBackend

SAMPLING = {"temperature": 0.0, "top_p": 1, "max_tokens": 1}   # greedy, one step


class VLLMSyncBackend(VLLMBackend):
    name = "vllm_sync"

    def __init__(self, dtype: str | None = None, trust_remote_code: bool = False,
                 max_model_len: int | None = None, tokenizer: str | None = None,
                 enable_prefix_caching: bool | None = None,
                 gpu_memory_utilization: float = 0.2, tensor_parallel_size: int = 1):
        super().__init__(dtype=dtype, trust_remote_code=trust_remote_code,
                         max_model_len=max_model_len, tokenizer=tokenizer,
                         enable_prefix_caching=enable_prefix_caching)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size

    def load(self, repo: str, gpu_memory_utilization: float | None = None):
        from nnsight.modeling.vllm import VLLM

        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
        kw = self._engine_kwargs()
        if self.tensor_parallel_size > 1:
            kw["tensor_parallel_size"] = self.tensor_parallel_size
        return VLLM(repo, mode="sync", dispatch=True,
                    gpu_memory_utilization=gpu_memory_utilization, **kw)

    def _collect(self, rows, n, what, keep_dtype=False):
        """Per-invoke outputs -> one tensor with the units along dim -2 (fp32 unless `keep_dtype`)."""
        import torch

        mats = [(r.detach() if keep_dtype else r.detach().float()).cpu() for r in rows if r is not None]
        if len(mats) != n:
            raise RuntimeError(f"vllm_sync {what} collected {len(mats)}/{n} per-unit outputs")
        return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)

    def run(self, model, prompts, build, *, keep_dtype=False):
        prompts = self.many(prompts, "run")
        with model.trace(**SAMPLING) as tracer:
            rows = [None] * len(prompts)
            rows = rows.save()                       # pre-sized slots: each invoke merges back its own
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    rows[i] = build([i])
        return self._collect(rows, len(prompts), "run", keep_dtype)

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        clean_prompts, corrupted_prompts = self.paired(clean_prompts, corrupted_prompts, "patch")
        n = len(clean_prompts)
        with model.trace(**SAMPLING) as tracer:     # trace 1: one invoke per clean prompt
            clean = [None] * n
            clean = clean.save()
            for i, p in enumerate(clean_prompts):
                with tracer.invoke(p):
                    clean[i] = capture()
        clean_acts = [c.detach().float().cpu() for c in clean]
        with model.trace(**SAMPLING) as tracer:     # trace 2: inject each into its corrupted run
            rows = [None] * n
            rows = rows.save()
            for i, p in enumerate(corrupted_prompts):
                with tracer.invoke(p):
                    rows[i] = patch(clean_acts[i])
        return self._collect(rows, n, "patch")

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        """Two traces of one invoke per pair: the source captures, then each pair's loss and its
        backward (`retain_graph=True` on all but the last: nnsight's form for backwards in several
        invokes of one trace); the engine's own grad refusal is the recorded verdict."""
        import torch

        sources, bases = self.paired(source_prompts, base_prompts, "train_patch")
        n = len(sources)
        with model.trace(**SAMPLING) as tracer:
            srcs = [None] * n
            srcs = srcs.save()
            for i, p in enumerate(sources):
                with tracer.invoke(p):
                    srcs[i] = capture()
        srcs = [s.detach().float().cpu() for s in srcs]
        with model.trace(**SAMPLING) as tracer:
            losses = [None] * n
            losses = losses.save()
            for i, p in enumerate(bases):
                with tracer.invoke(p):
                    loss = step(srcs[i], [i])
                    losses[i] = loss
                    with loss.backward(retain_graph=i < n - 1):
                        pass
        return torch.stack([l.detach().float().cpu().reshape(()) for l in losses]).sum()

    def vjp_batch(self, model, prompts, acts_of, target_of, make_cotangent, n):
        """One trace of one invoke per prompt, each taking its own backward (`retain_graph=True` on
        all but the last); the engine's own grad refusal is the recorded verdict. Returns the `n`
        per-layer gradients as `[B, seq, d]`."""
        import torch

        prompts = self.many(prompts, "vjp_batch")
        B = len(prompts)
        with model.trace(**SAMPLING) as tracer:
            grads = [None] * (B * n)
            grads = grads.save()                     # slot i * n + L: prompt i, layer L
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    acts = acts_of(model)
                    for a in acts:
                        a.requires_grad_(True)
                    target = target_of(model)
                    metric = (target * make_cotangent(target, [i])).sum()
                    with metric.backward(retain_graph=i < B - 1):
                        for L in range(n - 1, -1, -1):
                            grads[i * n + L] = acts[L].grad
        return [torch.stack([grads[i * n + L].detach().float().cpu() for i in range(B)])
                for L in range(n)]

    def attribute(self, model, clean_prompts, corrupt_prompts, acts_of, metric_of, n):
        """The same flow as HF's; the engine's own first grad refusal is the recorded verdict."""
        import torch

        pairs = list(zip(*self.paired(clean_prompts, corrupt_prompts, "attribute")))
        scores = []
        for i, (clean_prompt, corrupt_prompt) in enumerate(pairs):
            clean = [None] * n
            with model.trace(clean_prompt, **SAMPLING):
                a = acts_of(model)
                for L in range(n):
                    clean[L] = a[L].save()
            corrupt, grads = [None] * n, [None] * n
            with model.trace(corrupt_prompt, **SAMPLING):
                a = acts_of(model)
                for L in range(n):
                    a[L].requires_grad_(True)
                for L in range(n):
                    corrupt[L] = a[L].save()
                metric = metric_of(model, i)
                with metric.sum().backward():
                    for L in range(n - 1, -1, -1):
                        grads[L] = a[L].grad.save()
            scores.append(torch.stack([
                ((clean[L].float() - corrupt[L].float()) * grads[L].float()).sum().detach().cpu()
                for L in range(n)]))
        return torch.stack(scores)

    def _step_rows(self, per_unit, new_tokens, what):
        """One per-step list per unit -> `[new_tokens, B, ...]`."""
        import torch

        filled = [steps for steps in per_unit if steps is not None]
        if len(filled) != len(per_unit):
            raise RuntimeError(f"vllm_sync {what} collected {len(filled)}/{len(per_unit)} units")
        stacks = []
        for steps in per_unit:
            mats = [r.detach().float().cpu() for r in steps if r is not None]
            if len(mats) != new_tokens:
                raise RuntimeError(f"vllm_sync {what} collected {len(mats)} per-step rows, "
                                   f"expected {new_tokens}")
            stacks.append(torch.stack(mats, dim=0))                 # [new_tokens, 1, ...]
        return torch.cat(stacks, dim=1)

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        prompts = self.many(prompts, "generate")
        with model.trace(temperature=0.0, top_p=1, max_tokens=new_tokens,
                         min_tokens=new_tokens) as tracer:
            rows = [None] * len(prompts)
            rows = rows.save()                       # pre-sized slots: each invoke fills its own
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    per_step = []
                    rows[i] = per_step               # placed before the loop and filled in place
                    steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
                    for step in steps:
                        per_step.append(build_step(step))
        return self._step_rows(rows, new_tokens, "generation")

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        sources, bases = self.paired(source_prompts, base_prompts, "generate_patch")
        n = len(sources)
        with model.trace(**SAMPLING) as tracer:
            clean = [None] * n
            clean = clean.save()
            for i, p in enumerate(sources):
                with tracer.invoke(p):
                    clean[i] = capture()
        clean_acts = [c.detach().float().cpu() for c in clean]
        with model.trace(temperature=0.0, top_p=1, max_tokens=new_tokens,
                         min_tokens=new_tokens) as tracer:
            rows = [None] * n
            rows = rows.save()
            for i, p in enumerate(bases):
                with tracer.invoke(p):
                    per_step = []
                    rows[i] = per_step
                    steps = tracer.iter[0:new_tokens] if bounded else tracer.iter[:]
                    for step in steps:
                        per_step.append(build_step(clean_acts[i], step))
        return self._step_rows(rows, new_tokens, "generate_patch")

    def plain(self, model, units, *, new_tokens=1):
        """One `generate` call to nnsight's `vllm.LLM` entrypoint with no trace, which the engine
        schedules together."""
        from vllm import SamplingParams

        model.vllm_entrypoint.generate(
            prompts_of(units),
            SamplingParams(temperature=0.0, top_p=1, max_tokens=new_tokens, min_tokens=new_tokens),
            use_tqdm=False)

    def last(self, t):
        return t[-1:, :]                          # flat [tokens, vocab] -> [1, vocab]

    def collective_rpc(self, model, method, args=()):
        """Through `model.vllm_entrypoint`, the `vllm.LLM` nnsight runs (docs/models/vllm.md)."""
        return model.vllm_entrypoint.collective_rpc(method, args=args)

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

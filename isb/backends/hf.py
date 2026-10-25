"""HF Transformers backend: the per-family control. Several units run as one left-padded batch."""
from __future__ import annotations

from .base import Backend, prompts_of


class HFBackend(Backend):
    name = "hf"

    def __init__(self, force_text_causal: bool = False, dtype: str | None = None,
                 attn_implementation: str = "eager", device_map: str | None = None,
                 tp_size: int = 1):
        # force_text_causal: load through AutoModelForCausalLM for a text-only checkpoint of a
        # multimodal-wrapper architecture (Qwen/Qwen3.5-4B), which nnsight's guard otherwise refuses.
        self.force_text_causal = force_text_causal
        self.dtype = dtype          # None loads the checkpoint's own dtype
        self.attn_implementation = attn_implementation
        # None places the model on `load`'s one device; "auto" lets accelerate split its layers
        # across every visible GPU
        self.device_map = device_map
        # >1 shards every layer over that many torchrun ranks with transformers tensor parallelism
        self.tp_size = tp_size
        if tp_size > 1 and device_map is not None:
            raise ValueError("tp_size and device_map are mutually exclusive")

    def load(self, repo: str, device: str = "cuda:0"):
        import torch
        from nnsight import LanguageModel

        kw = {} if self.dtype is None else {"dtype": getattr(torch, self.dtype)}
        if self.force_text_causal:
            from transformers import AutoModelForCausalLM

            class _TextCausal(AutoModelForCausalLM):    # a non-default automodel: the guard opts out
                pass

            kw["automodel"] = _TextCausal
        if self.tp_size > 1:
            import os

            from transformers.distributed import DistributedConfig

            # this rank's GPU, so the timer's CUDA sync and peak memory read the device it runs on
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            kw["distributed_config"] = DistributedConfig(tp_size=self.tp_size)
        else:
            kw["device_map"] = self.device_map or device
        return LanguageModel(repo, dispatch=True, attn_implementation=self.attn_implementation, **kw)

    def run(self, model, prompts, build):
        prompts = self.many(prompts, "run")
        with model.trace(prompts):                       # the prompts as one left-padded batch
            saved = build(list(range(len(prompts)))).save()
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        clean_prompts, corrupted_prompts = self.paired(clean_prompts, corrupted_prompts, "patch")
        with model.trace(clean_prompts):                 # trace 1: snapshot the clean activations
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompts):             # trace 2: inject them, observe the batch
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def session(self, model, build):
        out = [None]                     # index-assign: the compiled session body shares the list
        with model.session():
            out[0] = build()
        return out[0]

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        sources, bases = self.paired(source_prompts, base_prompts, "train_patch")
        rows = list(range(len(sources)))
        with model.trace(sources):           # trace 1: source activations, no grad
            s = capture().save()
        src = s.detach().float().cpu()
        with model.trace(bases):             # trace 2: intervened forward + backward on the loss
            loss = step(src, rows)
            lv = loss.save()
            with loss.backward():
                pass                         # grads land on step's external trainables
        return lv.detach().float().cpu()

    def vjp_batch(self, model, prompts, acts_of, target_of, make_cotangent, n):
        prompts = self.many(prompts, "vjp_batch")
        grads = [None] * n
        with model.trace(prompts):
            a = acts_of(model)
            for L in range(n):
                a[L].requires_grad_(True)            # retain grad on the intermediate residuals
            target = target_of(model)
            cot = make_cotangent(target, list(range(len(prompts))))
            metric = (target * cot).sum()
            with metric.backward():
                for L in range(n - 1, -1, -1):       # grads in reverse module order
                    grads[L] = a[L].grad.save()
        return [g.detach().float().cpu() for g in grads]

    def attribute(self, model, clean_prompts, corrupt_prompts, acts_of, metric_of, n):
        """One pair per trace pair: the metric is a per-pair scalar, so pairs do not batch."""
        import torch

        pairs = list(zip(*self.paired(clean_prompts, corrupt_prompts, "attribute")))
        scores = []
        for i, (clean_prompt, corrupt_prompt) in enumerate(pairs):
            clean = [None] * n          # index-assign into lists made outside the compiled body
            with model.trace(clean_prompt):                  # pass 1: clean activations
                a = acts_of(model)
                for L in range(n):
                    clean[L] = a[L].save()
            corrupt, grads = [None] * n, [None] * n
            with model.trace(corrupt_prompt):                # pass 2: corrupt forward + backward
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
        return torch.stack(scores)                                   # [B, n]

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        prompts = self.many(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(prompts, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                            do_sample=False) as tracer:
            rows = list().save()
            for step in tracer.iter[it]:
                rows.append(build_step(step))
        return self._steps(rows, new_tokens, "generation")

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        sources, bases = self.paired(source_prompts, base_prompts, "generate_patch")
        with model.trace(sources):                  # trace 1: snapshot the source activations
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(bases, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                            do_sample=False) as tracer:
            rows = list().save()
            for step in tracer.iter[it]:
                rows.append(build_step(clean_act, step))
        return self._steps(rows, new_tokens, "generate_patch")

    def _steps(self, rows, new_tokens, what):
        """Per-step `[B, ...]` rows -> `[new_tokens, B, ...]`."""
        import torch

        if len(rows) != new_tokens:
            raise RuntimeError(f"hf {what} collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.stack([r.detach().float().cpu() for r in rows], dim=0)

    def plain(self, model, units, *, new_tokens=1):
        """The wrapped transformers module called directly on one left-padded batch, no trace. Ends
        where the cells end: the result copied to the host, so a timer sees the same boundary."""
        import torch

        inputs = model.tokenizer(prompts_of(units), return_tensors="pt", padding=True)
        inputs = inputs.to(model._module.device)
        with torch.no_grad():
            if new_tokens == 1:
                out = model._module(**inputs).logits[:, -1, :]
            else:
                out = model._module.generate(**inputs, max_new_tokens=new_tokens,
                                             min_new_tokens=new_tokens, do_sample=False)
        return out.cpu()

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

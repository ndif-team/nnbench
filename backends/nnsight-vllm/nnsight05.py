"""nnsight 0.5.15 (2026-01-13) on vLLM.

0.5 has one vLLM engine, the synchronous `vllm.LLM`, and its `VLLM` takes no `mode`: every keyword
it does not consume goes to vLLM's `EngineArgs` (modeling/vllm/vllm.py at v0.5.15, `_load`, which
also forces `enforce_eager=True`). Three forms differ from 0.7 and 0.8:

- one unit is traced as `model.trace(prompt, ...)`; 0.5 returns each invoke's saves by variable
  name, so a loop of invokes filling one container comes back empty and a call over several units
  is `Unsupported`;
- generation steps through `tracer.iter[...]` as a context manager binding the step index (its
  README: `with tracer.iter[:] as step:`);
- the model runner reads a mediator from every request's sampling params
  (model_runners/GPUModelRunner.py, `process_new_reqs`), so a plain request raises in the worker
  and takes the engine down: `plain` is unavailable.

backend.py selects it when nnsight 0.5 is installed.
"""
from isb.backends.vllm_sync import SAMPLING, VLLMSyncBackend
from isb.methodologies.registry import Unsupported

SEVERAL = ("nnsight 0.5's vLLM path returns each invoke's saves by variable name, so one call "
           "cannot gather a variable number of units")


class NNsight05SyncBackend(VLLMSyncBackend):
    def load(self, repo: str, gpu_memory_utilization: float | None = None):
        from nnsight.modeling.vllm import VLLM

        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
        kw = self._engine_kwargs()
        if self.tensor_parallel_size > 1:
            kw["tensor_parallel_size"] = self.tensor_parallel_size
        return VLLM(repo, dispatch=True, gpu_memory_utilization=gpu_memory_utilization, **kw)

    def one(self, prompts, primitive):
        """The single unit a 0.5 trace takes; several raise `Unsupported`."""
        prompts = self.many(prompts, primitive)
        if len(prompts) > 1:
            raise Unsupported(SEVERAL)
        return prompts[0]

    def run(self, model, prompts, build):
        prompt = self.one(prompts, "run")
        with model.trace(prompt, **SAMPLING):
            saved = build([0]).save()
        return saved.detach().float().cpu()

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        clean_prompts, corrupted_prompts = self.paired(clean_prompts, corrupted_prompts, "patch")
        clean_prompt = self.one(clean_prompts, "patch")
        with model.trace(clean_prompt, **SAMPLING):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        with model.trace(corrupted_prompts[0], **SAMPLING):
            res = patch(clean_act).save()
        return res.detach().float().cpu()

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        prompt = self.one(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=new_tokens,
                         min_tokens=new_tokens) as tracer:
            rows = list().save()
            with tracer.iter[it] as step:
                rows.append(build_step(step))
        return self._step_rows([rows], new_tokens, "generation")

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        sources, bases = self.paired(source_prompts, base_prompts, "generate_patch")
        source = self.one(sources, "generate_patch")
        with model.trace(source, **SAMPLING):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.trace(bases[0], temperature=0.0, top_p=1, max_tokens=new_tokens,
                         min_tokens=new_tokens) as tracer:
            rows = list().save()
            with tracer.iter[it] as step:
                rows.append(build_step(clean_act, step))
        return self._step_rows([rows], new_tokens, "generate_patch")

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        """One pair: the source capture, then the loss and its backward, each its own trace; the
        engine's own grad refusal is the recorded verdict. Several pairs are `Unsupported`."""
        sources, bases = self.paired(source_prompts, base_prompts, "train_patch")
        source = self.one(sources, "train_patch")
        with model.trace(source, **SAMPLING):
            s = capture().save()
        src = s.detach().float().cpu()
        with model.trace(bases[0], **SAMPLING):
            loss = step(src, [0])
            lv = loss.save()
            with loss.backward():
                pass
        return lv.detach().float().cpu().reshape(())

    def plain(self, model, units, *, new_tokens=1):
        raise NotImplementedError("nnsight 0.5's vLLM engine serves only traced requests")

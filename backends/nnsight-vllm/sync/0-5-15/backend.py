"""nnsight 0.5.15 (2026-01-13) on vLLM.

0.5 has one vLLM engine, the synchronous `vllm.LLM`, and its `VLLM` takes no `mode`: every keyword
it does not consume goes to vLLM's `EngineArgs` (nnsight modeling/vllm/vllm.py at v0.5.15, `_load`,
which also forces `enforce_eager=True`). The load leaves `mode` out. 0.5 steps through generation
with `tracer.iter[...]` as a context manager whose body runs once per step (its README:
`with tracer.iter[:]:`), so `generate` takes that form; the rest of the sync backend and the cells
apply unchanged (design.md §12.14). 0.5's model runner reads a mediator from every request's
sampling params (model_runners/GPUModelRunner.py, `process_new_reqs`), so a plain request through
its engine raises in the worker and takes the engine down: the plain-engine timing is unavailable.
"""
from isb.backends.vllm_sync import VLLMSyncBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


class NNsight05SyncBackend(VLLMSyncBackend):
    def load(self, repo: str, gpu_memory_utilization: float | None = None):
        from nnsight.modeling.vllm import VLLM

        if gpu_memory_utilization is None:
            gpu_memory_utilization = self.gpu_memory_utilization
        kw = self._engine_kwargs()
        if self.tensor_parallel_size > 1:
            kw["tensor_parallel_size"] = self.tensor_parallel_size
        return VLLM(repo, dispatch=True, gpu_memory_utilization=gpu_memory_utilization, **kw)

    def vanilla(self, model, prompt, *, new_tokens):
        raise NotImplementedError("nnsight 0.5's vLLM engine serves only traced requests")

    def vanilla_many(self, model, prompts, *, new_tokens):
        raise NotImplementedError("nnsight 0.5's vLLM engine serves only traced requests")

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        import torch

        prompt = self.one(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=new_tokens) as tracer:
            rows = list().save()
            with tracer.iter[it]:
                rows.append(build_step())
        if len(rows) != new_tokens:
            raise RuntimeError(
                f"vllm_sync generation collected {len(rows)} per-step rows, "
                f"expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)   # [steps, vocab]


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    return NNsight05SyncBackend(**params), RunConfig(EngineConfig("vllm", mode="sync", params={
        **params, "enforce_eager": True}))

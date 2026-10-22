"""nnsight 0.5.15 (2026-01-13) on HF transformers.

0.5 steps through generation with `tracer.iter[...]` as a context manager whose body runs once per
step (its README: `with tracer.iter[:]:`); 0.7 made it an iterator for a `for` loop. `generate`
takes 0.5's form; the rest of the HF backend applies unchanged.
"""
from isb.backends.hf import HFBackend
from isb.jobs.worker import merge_requirements, options
from isb.runs import EngineConfig, RunConfig


class NNsight05HFBackend(HFBackend):
    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        import torch

        prompt = self.one(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(prompt, max_new_tokens=new_tokens, do_sample=False) as tracer:
            rows = list().save()
            with tracer.iter[it]:
                rows.append(build_step())
        if len(rows) != new_tokens:
            raise RuntimeError(
                f"hf generation collected {len(rows)} per-step rows, expected {new_tokens}")
        return torch.cat([r.detach().float().cpu() for r in rows], dim=0)   # [steps, vocab]


def create_backend(spec):
    params = merge_requirements(spec.hf_kwargs, options())
    return NNsight05HFBackend(**params), RunConfig(EngineConfig("transformers", params=params))

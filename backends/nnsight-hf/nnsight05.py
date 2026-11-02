"""nnsight 0.5.15 (2026-01-13) on HF transformers.

0.5 steps through generation with `tracer.iter[...]` as a context manager whose body runs once per
step and binds the step index (its README: `with tracer.iter[:] as step:`); 0.7 made it an iterator
for a `for` loop. `generate` and `generate_patch` take 0.5's form; the rest of the HF backend
applies unchanged. backend.py selects it when nnsight 0.5 is installed.
"""
from isb.backends.hf import HFBackend


class NNsight05HFBackend(HFBackend):
    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        prompts = self.many(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(prompts, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                            do_sample=False) as tracer:
            rows = list().save()
            with tracer.iter[it] as step:
                rows.append(build_step(step))
        return self._steps(rows, new_tokens, "generation")

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        sources, bases = self.paired(source_prompts, base_prompts, "generate_patch")
        with model.trace(sources):
            ca = capture().save()
        clean_act = ca.detach().float().cpu()
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(bases, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                            do_sample=False) as tracer:
            rows = list().save()
            with tracer.iter[it] as step:
                rows.append(build_step(clean_act, step))
        return self._steps(rows, new_tokens, "generate_patch")

"""nnsight 0.5.15 (2026-01-13) on HF transformers.

0.5 steps through generation with `tracer.iter[...]` as a context manager whose body runs once per
step and binds the step index (its README: `with tracer.iter[:] as step:`); 0.7 made it an iterator
for a `for` loop. `generate` and `generate_patch` take 0.5's form; the rest of the HF backend
applies unchanged. backend.py selects it when nnsight 0.5 is installed.
"""
from isb.backends.hf import HFBackend, generated, greedy


class NNsight05HFBackend(HFBackend):
    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True, keep_dtype=False,
                 tokens=False):
        """The steps run in an invoke over the prompts; with `tokens`, a second, empty invoke reads
        the ids `generate` returns (`model.generator.output`), where code still runs after an
        unbounded loop (0.5.15 README, "Iterating Over Generation Steps"). The prompts go to the
        first invoke: 0.5.15 compiles a tracer given positional input as one invoke and refuses
        another inside it."""
        prompts = self.many(prompts, "generate")
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(**greedy(new_tokens)) as tracer:
            with tracer.invoke(prompts):
                rows = list().save()
                if build_step is not None:
                    with tracer.iter[it] as step:
                        rows.append(build_step(step))
            if tokens:
                with tracer.invoke():
                    ids = model.generator.output.save()
        if tokens:
            return generated(ids, new_tokens)
        return self._steps(rows, new_tokens, "generation", keep_dtype)

    def generate_patch(self, model, source_prompts, base_prompts, capture, inject, build_step,
                       *, new_tokens, bounded=True, keep_dtype=False, tokens=False):
        sources, bases = self.paired(source_prompts, base_prompts, "generate_patch")
        with model.trace(sources):
            ca = capture().save()
        clean_act = ca.detach()                     # stays where the trace saved it
        it = slice(0, new_tokens) if bounded else slice(None)
        with model.generate(**greedy(new_tokens)) as tracer:
            with tracer.invoke(bases):
                rows = list().save()
                inject(clean_act)                   # outside the loop: the first forward
                if build_step is not None:
                    with tracer.iter[it] as step:
                        rows.append(build_step(step))
            if tokens:
                with tracer.invoke():
                    ids = model.generator.output.save()
        if tokens:
            return generated(ids, new_tokens)
        return self._steps(rows, new_tokens, "generate_patch", keep_dtype)

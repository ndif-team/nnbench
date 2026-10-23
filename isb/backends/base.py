"""Backend infrastructure `be` (design.md §12.3).

The trace body must live in the SAME frame as `with model.trace(...)` — this dev branch
captures/compiles the body, so splitting them across a generator (`@contextmanager` +
yield) makes the captured body empty and the run deadlocks. So `be.run` owns the
`with model.trace(...)` and calls the cell's `build()` closure inside it; the cell stays
explicit (its closure names the model's own modules).

The cell never chooses the engine form. A cell hands every primitive the units of one call and
the backend runs them in the form its engine documents: HF pads several prompts into one batch,
the async vLLM engine runs one request per trace, the sync engine one invoke per prompt. A
primitive that takes one unit per side (`patch`, `generate`, `generate_patch`, `attribute`)
checks that through `one()` and raises `FormError` on more, so a cell that passes a list where
the engine takes one prompt fails in every backend and in the no-GPU test fakes, not only on vLLM.
`run` takes several prompts; its `build()` closure takes no arguments and the backend runs it in
its form (HF: one padded batch; vLLM: one invoke per prompt). `train_patch` and `vjp_batch` take
several units and tell the cell's closure which units the current trace holds (`rows`) so the
closure can pick the matching targets.
"""
from __future__ import annotations


class FormError(TypeError):
    """A cell handed a primitive more units than the primitive takes. A harness error: the cell,
    not the engine, chose a form; fix the cell."""


class Backend:
    name = "base"

    def one(self, prompt, primitive):
        """The single unit a one-unit primitive takes: a prompt, or a list holding one. More
        raise `FormError`. Every backend and test fake routes its one-unit primitives through
        this, so the form is checked before any engine sees it."""
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise FormError(
                    f"{self.name}.{primitive} takes one unit per side; the cell passed "
                    f"{len(prompt)}. Pass one unit per call, or use a primitive that takes several "
                    f"(run, train_patch, vjp_batch) and let the backend pick the engine's form.")
            return prompt[0]
        return prompt

    def many(self, prompts):
        """The unit list a several-unit primitive takes: a list as given, one prompt as a list of
        one."""
        return list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]

    def paired(self, sources, bases, primitive):
        """Both sides of a several-pair primitive as lists (`many()`); unequal sides raise
        `FormError`, since a source without a base has no pair to run."""
        sources, bases = self.many(sources), self.many(bases)
        if len(sources) != len(bases):
            raise FormError(f"{self.name}.{primitive} takes one base per source; the cell passed "
                            f"{len(sources)} source and {len(bases)} base prompts.")
        return sources, bases

    def load(self, repo: str, **kw):
        raise NotImplementedError

    def run(self, model, prompts, build):
        """Open the trace, call `build()` (-> a proxy) inside it, save, return CPU tensor."""
        raise NotImplementedError

    def patch(self, model, clean_prompt, corrupted_prompt, capture, patch):
        """Two-trace activation patching: capture a value from a CLEAN single-prompt trace, then
        inject it in a CORRUPTED single-prompt trace and return the corrupted run's CPU result.

        `capture()` runs inside trace 1 and returns the proxy to snapshot. `patch(clean_cpu_tensor)`
        runs inside trace 2 (the captured value is a materialized CPU tensor by then) and returns the
        observed proxy. One unit per side (`one()`): two single-prompt traces need no
        multi-invoke/barrier, which is why this runs on vLLM, where continuous-batch multi-invoke
        is unsupported. A cell with several pairs calls this once per pair.
        """
        raise NotImplementedError

    def session(self, model, build):
        """Run `build()`, a job of several traces with Python between them (a training loop, a
        collection over prompts), in the form nnsight documents for a multi-trace job: one
        `model.session()` around it. Returns what `build()` returns."""
        raise NotImplementedError

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        """One gradient step over the (source, base) pairs `zip(source_prompts, base_prompts)` (a
        single prompt per side is one pair; unequal sides raise `FormError`, `paired()`). The
        backend runs the pairs in its engine's form: HF pads each side into one batch, two traces
        in all; the async vLLM engine runs one pair per pair of traces and accumulates the
        gradients. Trace 1 captures the source activation
        (materialized to CPU); trace 2 calls `step(src, rows)` -> a scalar loss proxy inside the
        trace and runs its backward. `src` holds the captured activations of the pairs `rows`
        (indices into the lists passed, in the backend's batch layout), so `step` selects the
        targets of those pairs. `step` returns the SUM of the per-pair losses of `rows` (a cell
        that wants the mean divides each by the number of pairs it passed), so the gradient and
        the returned loss are the same whether the backend runs the pairs in one trace or one
        pair per trace. Gradients land on any external trainable tensors `step` closes over (the
        DAS rotation); the CALLER owns the optimizer (step/zero outside). Returns the loss summed
        over all pairs as a CPU scalar tensor."""
        raise NotImplementedError

    def vjp_batch(self, model, prompts, acts_of, target_of, make_cotangent, n):
        """One VJP sweep over `prompts`, in the engine's form (HF: one padded batch, one trace;
        the async vLLM engine: one trace per prompt). Inside each trace `target_of(model)` is the
        tensor the cotangent contracts with, `make_cotangent(target, rows)` builds the cotangent
        for the prompts `rows` (indices into `prompts`, one one-hot output dimension per row), and
        the backward of `(target * cot).sum()` populates grads on the `n` activation proxies from
        `acts_of(model)`. Returns the `n` per-layer gradients as CPU tensors in the batch layout
        (`[len(prompts), seq, d]`, or `[seq, d]` for one prompt on a flat engine)."""
        raise NotImplementedError

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n):
        """Attribution patching: a first-order linear approximation of activation patching over a
        clean/corrupt pair. `acts_of(model)` returns the `n` per-layer activation proxies to attribute
        (called fresh inside each trace, in forward order); `metric_of(model)` returns the scalar
        metric proxy; `n` is the layer count (passed in so the result lists can be created outside the
        trace). Returns a CPU `[n]` tensor of `((act_clean - act_corrupt) * grad).sum()` per layer,
        where `grad = d(metric)/d(act)` on the corrupt run. One prompt per side (`one()`).

        This needs autograd — a clean forward plus a corrupt forward+backward. It is the backend's
        gradient primitive: HF supports it; vLLM runs in inference mode (its activations are inference
        tensors with no autograd), so it cannot, and surfaces that as a per-cell ERROR.
        """
        raise NotImplementedError

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        """Greedy multi-token decode with a per-step intervention: open the generation trace,
        iterate the decode steps, call `build_step()` once per step inside the loop (it intervenes
        and returns that step's per-step proxy, e.g. last-token logits), and return the stacked
        per-step CPU tensor `[new_tokens, ...]`.

        `bounded` selects the iteration REALIZATION (Level 1.5): True = `tracer.iter[0:N]`
        (carries its own stop bound — the working idiom on vLLM); False = `tracer.iter[:]`
        (relies on the engine-supplied stop bound — works on HF, drops ALL saves on vLLM).
        One prompt (`one()`): generation is the per-trace decode regime, not a batching axis.
        """
        raise NotImplementedError

    def generate_patch(self, model, source_prompt, base_prompt, capture, build_step,
                       *, new_tokens, bounded=True):
        """Cross-prompt transplant run UNDER the decode loop = the two-trace patch (§be.patch)
        composed with the generation loop (§generate). `capture()` runs in trace 1 (a single
        forward on `source_prompt`) and returns the proxy to snapshot; `build_step(clean_cpu)` runs
        once per decode step of `base_prompt` in trace 2 (the captured value is a materialized CPU
        tensor by then) — it injects at the prompt positions during the PREFILL step (where the
        full-prompt residual matches the captured shape) and reads that step's per-step proxy.

        This is a COMPOSITION cell: it exercises no new primitive — the boundary read, the
        replacement write, the two-trace transplant edge, and bounded iteration are each already
        measured. Its job is to check the step-lift LAW for the transplant edge (does a transplant
        stay valid run during a decode loop), combining cross-prompt interchange with generated
        token observations. Returns the stacked
        per-step CPU tensor `[new_tokens, ...]`.

        Note on the patch persisting across decode steps: the injection lands at PREFILL; decode
        steps don't recompute prompt positions, so there is nothing to re-inject — the patched
        residual is simply part of the forward that builds this request's (intra-request) KV cache.
        No cross-request prefix cache is involved (nnsight disables it), so this is ordinary
        dataflow, not a separate propagation that could fail.
        """
        raise NotImplementedError

    def vanilla(self, model, prompt, *, new_tokens):
        """The system's own request with no intervention attached: one forward over `prompt`
        (`new_tokens=1`) or a greedy generation of `new_tokens`. Its latency is the denominator for
        intervention overhead (design.md §12.14). Returns nothing; only its timing is recorded."""
        raise NotImplementedError(f"{self.name} has no plain request")

    def last(self, t):
        """Last-token row of a logits tensor (backend-shape-specific)."""
        raise NotImplementedError

    def teardown(self, model) -> None:
        pass

"""Backend infrastructure `be` (design.md §12.3).

The trace body must live in the SAME frame as `with model.trace(...)` — this dev branch
captures/compiles the body, so splitting them across a generator (`@contextmanager` +
yield) makes the captured body empty and the run deadlocks. So `be.run` owns the
`with model.trace(...)` and calls the cell's `build()` closure inside it; the cell stays
explicit (its closure names the model's own modules).
"""
from __future__ import annotations


class Backend:
    name = "base"

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
        observed proxy. Both prompts are SINGLE prompts, so this never needs multi-invoke/barrier —
        which is exactly why it can run on vLLM, where continuous-batch multi-invoke is unsupported.
        """
        raise NotImplementedError

    def attribute(self, model, clean_prompt, corrupt_prompt, acts_of, metric_of, n):
        """Attribution patching: a first-order linear approximation of activation patching over a
        clean/corrupt pair. `acts_of(model)` returns the `n` per-layer activation proxies to attribute
        (called fresh inside each trace, in forward order); `metric_of(model)` returns the scalar
        metric proxy; `n` is the layer count (passed in so the result lists can be created outside the
        trace). Returns a CPU `[n]` tensor of `((act_clean - act_corrupt) * grad).sum()` per layer,
        where `grad = d(metric)/d(act)` on the corrupt run.

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
        Single prompt: generation is the per-trace decode regime, not a batching axis.
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
        stay valid run during a decode loop), and to be the recipe a causalab `locate`-style
        analysis needs (cross-prompt interchange scored on generated tokens). Returns the stacked
        per-step CPU tensor `[new_tokens, ...]`.

        Note on the patch persisting across decode steps: the injection lands at PREFILL; decode
        steps don't recompute prompt positions, so there is nothing to re-inject — the patched
        residual is simply part of the forward that builds this request's (intra-request) KV cache.
        No cross-request prefix cache is involved (nnsight disables it), so this is ordinary
        dataflow, not a separate propagation that could fail.
        """
        raise NotImplementedError

    def last(self, t):
        """Last-token row of a logits tensor (backend-shape-specific)."""
        raise NotImplementedError

    def teardown(self, model) -> None:
        pass

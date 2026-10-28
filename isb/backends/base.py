"""Backend infrastructure `be`: the primitives a cell hands its trace closures to.

A cell hands every primitive the units of one call, always as a list (`prompts`, or one list per
side of a pair), and the backend runs them in the form its engine documents: HF pads several
prompts into one batch, the sync vLLM engine runs one invoke per prompt in one trace, the async
engine one request per prompt, gathered. A trace closure runs in the same frame as the backend's
`with model.trace(...)`; nnsight compiles that frame, so the backend owns the `with` and calls the
closure inside it. Closures are named functions, never lambdas: under tensor or pipeline
parallelism nnsight source-serializes them to the worker.

Outputs carry the unit axis at dim -2 (`[B, vocab]`, `[L, B, vocab]`, `[T, B, vocab]`), so a
call over one unit returns the `[.., 1, ..]` slice of the same call over several.
"""
from __future__ import annotations


class FormError(TypeError):
    """A cell handed a primitive something other than the unit lists it takes."""


def prompts_of(units):
    """The prompts a list of units runs: a prompt unit is itself, a pair unit (clean, corrupt[,
    labels]) runs both sides."""
    out = []
    for unit in units:
        out.extend(unit[:2] if isinstance(unit, tuple) else [unit])
    return out


class Backend:
    name = "base"

    def many(self, prompts, primitive):
        """The unit list a primitive takes; anything but a nonempty list of units raises."""
        if not isinstance(prompts, list) or not prompts:
            raise FormError(f"{self.name}.{primitive} takes a nonempty list of units, got "
                            f"{type(prompts).__name__}")
        return prompts

    def paired(self, sources, bases, primitive):
        """Both sides of a pairwise primitive as equally long lists."""
        sources, bases = self.many(sources, primitive), self.many(bases, primitive)
        if len(sources) != len(bases):
            raise FormError(f"{self.name}.{primitive} takes one base per source; the cell passed "
                            f"{len(sources)} source and {len(bases)} base prompts.")
        return sources, bases

    def load(self, repo: str, **kw):
        raise NotImplementedError

    def run(self, model, prompts, build, *, keep_dtype=False):
        """One forward over `prompts`. `build(rows)` runs inside the trace with the indices (into
        `prompts`) of the units that trace holds and returns the proxy to save. Returns the saved
        value as a CPU tensor with the units along dim -2: in fp32, or with `keep_dtype` in the
        dtype the trace saved (an exported activation, whose volume is part of the measurement)."""
        raise NotImplementedError

    def patch(self, model, clean_prompts, corrupted_prompts, capture, patch):
        """Two-trace activation patching over pairs: `capture()` runs in the clean trace and returns
        the proxy to snapshot; `patch(clean)` runs in the corrupted trace with that snapshot as a CPU
        tensor (the trace's own units) and returns the proxy to observe."""
        raise NotImplementedError

    def session(self, model, build):
        """Run `build()`, a job of several traces with Python between them, in the form nnsight
        documents for a multi-trace job. Returns what `build()` returns."""
        raise NotImplementedError

    def train_patch(self, model, source_prompts, base_prompts, capture, step):
        """One gradient step over the pairs `zip(source_prompts, base_prompts)`: trace 1 captures
        the source activation (to CPU), trace 2 calls `step(src, rows)` for the loss over the pairs
        `rows` the trace holds (summed over them) and runs its backward. Returns the loss summed
        over all pairs as a CPU scalar."""
        raise NotImplementedError

    def vjp_batch(self, model, prompts, acts_of, target_of, make_cotangent, n):
        """One VJP sweep over `prompts`: inside each trace `make_cotangent(target_of(model), rows)`
        builds the cotangent for the units `rows` the trace holds and the backward of
        `(target * cot).sum()` populates grads on the `n` proxies from `acts_of(model)`. Returns
        the `n` per-layer gradients as CPU tensors in the batch layout."""
        raise NotImplementedError

    def attribute(self, model, clean_prompts, corrupt_prompts, acts_of, metric_of, n):
        """Attribution patching over pairs: `acts_of(model)` returns the `n` activation proxies,
        `metric_of(model, i)` the scalar metric proxy of pair `i`. Returns a CPU `[B, n]` tensor of
        `((act_clean - act_corrupt) * grad).sum()` per pair and layer. Needs autograd."""
        raise NotImplementedError

    def generate(self, model, prompts, build_step, *, new_tokens, bounded=True):
        """Greedy decode of `new_tokens` steps over `prompts`; `build_step(step)` runs once per
        decode step and returns that step's proxy (`[B_trace, ...]`). Returns `[new_tokens, B, ...]`.
        `bounded` selects `tracer.iter[0:N]` (True) or `tracer.iter[:]` (False)."""
        raise NotImplementedError

    def generate_patch(self, model, source_prompts, base_prompts, capture, build_step,
                       *, new_tokens, bounded=True):
        """`patch` composed with `generate`: `capture()` runs in a forward over the sources,
        `build_step(clean, step)` once per decode step of the bases with the snapshot as a CPU
        tensor. Returns `[new_tokens, B, ...]`."""
        raise NotImplementedError

    def plain(self, model, units, *, new_tokens=1):
        """The system's own request with no intervention attached over `units` (`prompts_of`): one
        forward when `new_tokens` is 1, else a greedy generation. Only its timing is recorded."""
        raise NotImplementedError(f"{self.name} has no plain request")

    def last(self, t):
        """Last-token row of a per-position tensor (residual stream or logits)."""
        raise NotImplementedError

    def collective_rpc(self, model, method, args=()):
        """Run `method(worker, *args)` in every engine worker process through the engine's own
        `collective_rpc` and return the per-rank results. Only a backend whose engine runs in
        separate worker processes (vLLM) has one; the harness reads the workers' peak memory with
        it (isb/perf/worker_memory.py). A backend that does not override this records why."""
        raise NotImplementedError(f"{self.name} gives the harness no handle on engine worker "
                                  f"processes (no collective_rpc override)")

    def teardown(self, model) -> None:
        pass

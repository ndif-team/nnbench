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

    def last(self, t):
        """Last-token row of a logits tensor (backend-shape-specific)."""
        raise NotImplementedError

    def teardown(self, model) -> None:
        pass

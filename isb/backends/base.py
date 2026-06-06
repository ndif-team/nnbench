"""Backend (system-under-test) interface (design.md §6, §7, §11.8).

A backend knows how to (1) instantiate the model on a serving stack, (2) execute a
motif Program inside a trace and collect a single saved tensor, and (3) tear down.

`BackendCtx` injects the two shape-sensitive ops a motif needs, so motif code stays
backend-agnostic: HF activations are [B, S, H]; vLLM's continuous-batching flat buffer
is [total_tokens, H] (OSDI optimization axis), so `select_last` differs per backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BackendCtx:
    select_last: Callable[[Any], Any]  # pick the last-token row, returns [1, vocab]
    stack: Callable[[list], Any]       # stack per-site rows -> [n_sites, 1, vocab]


class Backend:
    name = "base"

    def load(self, repo: str, **kw):
        raise NotImplementedError

    def run(self, model, program, prompt: str, generation) -> dict:
        """Return {"value": cpu_tensor, "site_ids": [...], "latency_s": float}."""
        raise NotImplementedError

    def teardown(self, model) -> None:
        pass

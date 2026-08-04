"""Logit lens: family-generic cell (design.md §12.1, §12.8).

The lens intervention is identical across families; only WHERE the blocks / final norm /
head live differs, and that is the profile's job (`m`, isb/profiles.py). One hf + one
vllm_async cell registered `family="*"` replace the six per-family copies; the registry
binds the spec's family string to its ModelProfile at lookup. An explicit per-family cell
would still take precedence if a family ever needs different lens code; none does today
(gpt2 / llama-tree / nemotron differ only in paths and residual denotation, both profile
rows).

Variances (§12, runtime params): `prompts` (list -> batched by `be.run`); `layers`
("all" | list[int], interpreted against the profile's block list); `unembed` ("module"
idiomatic | "weight" portable: vLLM guards `lm_head.forward`, so the portable form uses
the weight matmul); `residual` ("plain" | "fused", see profiles._resid; the vLLM default
derives from the family's denotation, and passing residual="plain" explicitly remains the
naive-port frontier task on fused-residual families).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .registry import cell


def _lens_proxy(blocks, norm, head, *, layers, unembed, last_fn, residual="plain"):
    """Build the stacked logit-lens proxy. Runs INSIDE a trace (no trace-open here)."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():                                   # forward-only; required on vLLM, harmless on HF
        for i in idx:
            normed = norm(_resid(blocks[i].output, residual))  # residual stream -> final norm
            logits = (
                F.linear(normed, head.weight)           # portable: bypass lm_head.forward guard
                if unembed == "weight"
                else head(normed)                       # idiomatic: model's own head
            )
            rows.append(last_fn(logits))                # last-token row
    return torch.stack(rows, dim=0)                     # [n_layers, ., vocab]


@cell("logit_lens", family="*", backend="hf")
def logit_lens_hf(be, model, m, prompts, *, layers="all", unembed="module", residual="plain"):
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            m.blocks(model), m.norm(model), m.head(model),
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)


@cell("logit_lens", family="*", backend="vllm_async")
def logit_lens_vllm(be, model, m, prompts, *, layers="all", unembed="weight", residual=None):
    # The residual default derives from the family's vLLM denotation: fused-residual families must
    # read hidden+residual (a plain read drops the accumulated residual -> SILENTLY_WRONG, the
    # fused-residual denotation mismatch). An explicit residual="plain" still overrides: that is the
    # naive-port frontier task the llama spec keeps as a marker.
    residual = residual if residual is not None else m.default_residual(be.name)

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            m.blocks(model), m.norm(model), m.head(model),
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)

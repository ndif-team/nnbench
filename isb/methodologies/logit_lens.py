"""Logit lens: family-generic cell (design.md §12.1, §12.8).

The lens intervention is identical across families; only WHERE the blocks / final norm /
head live differs, and that is the profile's job (`m`, isb/profiles.py). One hf + one
vllm_async cell registered `family="*"` replace the six per-family copies; the registry
binds the spec's family string to its ModelProfile at lookup. An explicit per-family cell
would still take precedence if a family ever needs different lens code; none does today
(gpt2 / llama-tree / nemotron differ only in paths and residual denotation, both profile
rows).

Variances (§12, runtime params): `prompts` (list -> batched by `be.run`); `layers`
("all" | list[int], interpreted against the profile's block list); `unembed` ("module": the
model's head called on the normed stream, the HF form of docs/patterns/logit-lens.md;
"logits_processor": `model.logits_processor(model.lm_head, normed)`, the vLLM form of
docs/models/vllm-parallelism.md, which gathers the vocabulary shards under tensor parallelism
(`lm_head.forward` itself is guarded on vLLM); "weight": a matmul with `lm_head.weight`, which is
only this rank's slice under tensor parallelism, kept as an alternative); `residual` ("plain" |
"fused", see profiles._resid; the vLLM default
derives from the family's denotation, and passing residual="plain" explicitly remains the
naive-port frontier task on fused-residual families).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .observations import record_resolved
from .registry import cell


def _lens_proxy(blocks, norm, head, *, layers, unembed, last_fn, residual="plain", processor=None):
    """Build the stacked logit-lens proxy. Runs INSIDE a trace (no trace-open here)."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():                                   # forward-only; required on vLLM, harmless on HF
        for i in idx:
            # last token first, then the final norm and unembed: both act per position, so this is
            # the full-sequence read's last row without unembedding (and holding) every position
            normed = norm(last_fn(_resid(blocks[i].output, residual)))
            if unembed == "module":
                rows.append(head(normed))               # the model's own head (HF)
            elif unembed == "logits_processor":
                rows.append(processor(head, normed))    # vLLM's logits path, whole vocabulary
            elif unembed == "weight":
                rows.append(F.linear(normed, head.weight))   # this rank's slice under TP
            else:
                raise ValueError(f"unknown unembed {unembed!r}")
    return torch.stack(rows, dim=0)                     # [n_layers, ., vocab]


@cell("logit_lens", family="*", backend="hf", realizations=[{"unembed": "weight"}])
def logit_lens_hf(be, model, m, prompts, *, layers="all", unembed="module", residual="plain"):
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            m.blocks(model), m.norm(model), m.head(model),
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)


# alternatives: the weight-matmul unembed (a rank slice under TP), and the naive port (plain
# residual read; SILENTLY_WRONG on
# fused-residual families)
@cell("logit_lens", family="*", backend="vllm_async",
      realizations=[{"unembed": "weight"}, {"residual": "plain"}])
def logit_lens_vllm(be, model, m, prompts, *, layers="all", unembed="logits_processor",
                    residual=None):
    # The residual default derives from the family's vLLM denotation: fused-residual families must
    # read hidden+residual (a plain read drops the accumulated residual -> SILENTLY_WRONG, the
    # fused-residual denotation mismatch). An explicit residual="plain" still overrides: that is the
    # naive-port alternative this cell declares, kept as a frontier marker.
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            m.blocks(model), m.norm(model), m.head(model),
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
            processor=model.logits_processor,
        )
    return be.run(model, prompts, build)

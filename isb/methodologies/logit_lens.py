"""Logit lens: the final norm and unembedding applied to each block's last-token residual stream.
Family-generic; the profile `m` says where the blocks, norm and head live. `unembed` selects the
projection: the model's head module (HF), vLLM's `logits_processor` (whole under tensor
parallelism) or a matmul with the head weight (this rank's slice under TP). `residual` selects the
stream read on vLLM's fused-residual families (`fused` = hidden + residual)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .observations import record_resolved
from .registry import cell


def _lens_proxy(blocks, norm, head, *, layers, unembed, last_fn, residual="plain", processor=None):
    """The stacked logit-lens proxy `[n_layers, B, vocab]`. Runs inside a trace."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():
        for i in idx:
            normed = norm(last_fn(_resid(blocks[i].output, residual)))
            if unembed == "module":
                rows.append(head(normed))
            elif unembed == "logits_processor":
                rows.append(processor(head, normed))
            elif unembed == "weight":
                rows.append(F.linear(normed, head.weight))
            else:
                raise ValueError(f"unknown unembed {unembed!r}")
    return torch.stack(rows, dim=0)


@cell("logit_lens", family="*", backend="hf", realizations=[{"unembed": "weight"}])
def logit_lens_hf(be, model, m, units, *, layers="all", unembed="module", residual="plain"):
    def build(rows):
        return _lens_proxy(m.blocks(model), m.norm(model), m.head(model), layers=layers,
                           unembed=unembed, last_fn=be.last, residual=residual)

    return be.run(model, units, build)


# alternatives: the weight-matmul unembed (a rank slice under TP) and the plain-residual read, the
# naive port that drops the accumulated residual on fused-residual families
@cell("logit_lens", family="*", backend="vllm_async",
      realizations=[{"unembed": "weight"}, {"residual": "plain"}])
def logit_lens_vllm(be, model, m, units, *, layers="all", unembed="logits_processor", residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def build(rows):
        return _lens_proxy(m.blocks(model), m.norm(model), m.head(model), layers=layers,
                           unembed=unembed, last_fn=be.last, residual=residual,
                           processor=m.logits_processor(model))

    return be.run(model, units, build)

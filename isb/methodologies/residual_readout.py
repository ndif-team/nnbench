"""Residual readout: the residual stream after each listed block at each prompt's last token, copied
to host memory, `[len(layers), B, d]` in the engine's activation dtype. HF's block output is the
stream itself and `be.last` takes each padded row's last (real, under left padding) position; vLLM's
fused-residual decoder layers return `(hidden, residual)` and the stream after the layer is their
sum (`residual="fused"`, the family's vLLM default). Blocks are read in forward order."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .registry import cell


@cell("residual_readout", family="*", backend="hf")
def residual_readout_hf(be, model, m, units, *, layers, residual="plain"):
    def build(rows):
        blocks = m.blocks(model)
        return torch.stack([be.last(_resid(blocks[i].output, residual)) for i in layers])

    with torch.no_grad():
        return be.run(model, units, build, keep_dtype=True)            # [n_layers, B, d]


@cell("residual_readout", family="*", backend="vllm_async")
def residual_readout_vllm(be, model, m, units, *, layers, residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def build(rows):
        blocks = m.blocks(model)
        # a fused read sums into a fresh tensor, which the next layer's in-place fused add-norm
        # cannot overwrite; a plain read is GPT-2's block output, which no later op writes
        return torch.stack([be.last(_resid(blocks[i].output, residual)) for i in layers]).cpu()

    return be.run(model, units, build, keep_dtype=True)                # [n_layers, B, d]

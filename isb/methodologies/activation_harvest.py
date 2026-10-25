"""Activation harvesting: the residual stream at every prompt token, copied to host memory,
`[tokens, d]` in prompt order. HF's batch is left-padded to its longest prompt, so prompt i's tokens
are the last n_i positions of row i and the pad positions are dropped; vLLM serves each request's
own `[tokens, d]` with no padding. `export=False` is the baseline: the same forward, returning each
prompt's greedy next token."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import engine_logits
from .registry import cell


@cell("activation_harvest", family="*", backend="hf")
def activation_harvest_hf(be, model, m, units, *, layer, export=True, residual="plain"):
    counts = [len(model.tokenizer(p)["input_ids"]) for p in units]
    if len(units) > 1 and model.tokenizer.padding_side != "left":
        raise ValueError(f"the tokenizer pads on the {model.tokenizer.padding_side}; valid tokens are "
                         f"located as each row's last positions, which needs left padding")

    def build(rows):
        if not export:
            return be.last(m.head(model).output).argmax(-1, keepdim=True).cpu()     # [B, 1]
        h = _resid(m.blocks(model)[layer].output, residual)                        # [B, seq, d]
        if h.shape[1] != max(counts):
            raise ValueError(f"batch is padded to {h.shape[1]} positions but the longest prompt "
                             f"tokenizes to {max(counts)}; valid tokens cannot be located")
        return torch.cat([row[row.shape[0] - counts[r]:] for row, r in zip(h, rows)]).cpu()

    with torch.no_grad():
        return be.run(model, units, build)


@cell("activation_harvest", family="*", backend="vllm_async")
def activation_harvest_vllm(be, model, m, units, *, layer, export=True, residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def build(rows):
        if not export:
            return engine_logits(model)[-1:, :].argmax(-1, keepdim=True).cpu()      # [1, 1]
        return _resid(m.blocks(model)[layer].output, residual).cpu()               # [tokens, d]

    return be.run(model, units, build)

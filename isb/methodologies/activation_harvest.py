"""Activation harvesting: the residual stream at every prompt token, copied to host memory,
`[tokens, d]` in prompt order. HF's batch is left-padded to its longest prompt, so prompt i's tokens
are the last n_i positions of row i and the pad positions are dropped; vLLM serves each request's
own `[tokens, d]` with no padding. `export=False` is the baseline: the same forward, returning each
prompt's greedy next token. `layers`, when given, replaces `layer`: the result is `[len(layers),
tokens, d]`, each layer's rows in the single-layer order, in the engine's activation dtype."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import engine_logits
from .registry import cell


def _layers(layer, layers, export):
    """The layers a call reads (None for the export=False baseline, which reads none)."""
    if not export:
        return None
    if layers is None and layer is None:
        raise ValueError("activation_harvest needs `layer` or `layers`")
    return None if layers is None else list(layers)


@cell("activation_harvest", family="*", backend="hf")
def activation_harvest_hf(be, model, m, units, *, layer=None, export=True, residual="plain",
                          layers=None):
    counts = [len(model.tokenizer(p)["input_ids"]) for p in units]
    if len(units) > 1 and model.tokenizer.padding_side != "left":
        raise ValueError(f"the tokenizer pads on the {model.tokenizer.padding_side}; valid tokens are "
                         f"located as each row's last positions, which needs left padding")
    read = _layers(layer, layers, export)

    def valid(h, rows):
        """A padded `[B, seq, d]` stream -> its valid tokens `[tokens, d]`, prompt by prompt."""
        if h.shape[1] != max(counts):
            raise ValueError(f"batch is padded to {h.shape[1]} positions but the longest prompt "
                             f"tokenizes to {max(counts)}; valid tokens cannot be located")
        return torch.cat([row[row.shape[0] - counts[r]:] for row, r in zip(h, rows)])

    def build(rows):
        if not export:
            return be.last(m.head(model).output).argmax(-1, keepdim=True).cpu()     # [B, 1]
        if read is None:
            return valid(_resid(m.blocks(model)[layer].output, residual), rows).cpu()   # [tokens, d]
        blocks = m.blocks(model)
        return torch.stack([valid(_resid(blocks[i].output, residual), rows)
                            for i in read])                                # [n_layers, tokens, d]

    with torch.no_grad():
        if read is None:
            return be.run(model, units, build)
        return be.run(model, units, build, keep_dtype=True)


@cell("activation_harvest", family="*", backend="vllm_async")
def activation_harvest_vllm(be, model, m, units, *, layer=None, export=True, residual=None,
                            layers=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    read = _layers(layer, layers, export)

    def build(rows):
        if not export:
            return engine_logits(model)[-1:, :].argmax(-1, keepdim=True).cpu()      # [1, 1]
        if read is None:
            return _resid(m.blocks(model)[layer].output, residual).cpu()           # [tokens, d]
        blocks = m.blocks(model)
        return torch.stack([_resid(blocks[i].output, residual)
                            for i in read]).cpu()                          # [n_layers, tokens, d]

    if read is None:
        return be.run(model, units, build)
    return be.run(model, units, build, keep_dtype=True)

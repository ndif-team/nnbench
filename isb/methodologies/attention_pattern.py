"""Attention-pattern read: the post-softmax attention probabilities of the last query token, as
log-probabilities so the oracle's softmax recovers them, `[n_layers, B, n_heads, k_len]`. HF eager
attention materializes them as `attn.source.attention_interface_0.output[1]`; vLLM's paged
attention never does, so the read raises there (the recorded verdict)."""
from __future__ import annotations

import torch

from .registry import cell


def _attn_pattern(blocks, *, layers):
    """Stack the last-query attention pattern (log-probs) over the requested layers. Runs inside a
    trace."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():
        for i in idx:
            weights = blocks[i].attn.source.attention_interface_0.output[1]   # [B, heads, q, k]
            rows.append(torch.log(weights[:, :, -1, :]))                      # [B, heads, k]
    return torch.stack(rows, dim=0)


@cell("attention_pattern", family="gpt2", backend="hf")
def attention_pattern_gpt2_hf(be, model, units, *, layers="all"):
    def build(rows):
        return _attn_pattern(model.transformer.h, layers=layers)

    return be.run(model, units, build)


@cell("attention_pattern", family="gpt2", backend="vllm_async")
def attention_pattern_gpt2_vllm(be, model, units, *, layers="all"):
    def build(rows):
        return _attn_pattern(model.transformer.h, layers=layers)

    return be.run(model, units, build)

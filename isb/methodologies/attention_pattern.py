"""Attention-pattern read — read the post-softmax attention probability matrix (design.md §12).

The attention pattern `A = softmax(QK^T / sqrt(d))` is the most direct read on what each head
attends to. A block's `.output` returns the *value-weighted* result, NOT the probabilities, so the
weights are reached via `.source` (intermediate ops inside the attention forward, see nnsight
docs/usage/source.md). For HF eager attention the call is `attn_output, attn_weights =
attention_interface(...)`, exposed as `attn.source.attention_interface_0.output` (a 2-tuple); element
`[1]` is `attn_weights` shaped `[batch, n_heads, q_len, k_len]`.

**Backend frontier — this cell exercises the `attn-weights` primitive.** HF eager materializes the
probability matrix; vLLM's paged/flash attention computes attention implicitly and never materializes
it (and runs a different forward with no `attention_interface` op), so the read raises -> ERROR. That
HF-vs-vLLM split is the finding.

We emit `log(A)` for the LAST query token so the equivalence oracle can be reused verbatim: the last
token attends to every key under the causal mask, so each entry is strictly positive and the log is
finite, and `softmax(log p) = p` for a row that already sums to 1 — so the oracle's softmax recovers
the true attention distribution (top-1 = the key each head attends to most, TV = the true
attention-distribution difference). Requires `attn_implementation="eager"` on HF (the HF backend
loads that way).

Variance (param, §12): `layers` ("all" | list[int], the observe set, indexed into the model's own
block list).
"""
from __future__ import annotations

import torch

from .registry import cell


def _attn_pattern(blocks, *, layers):
    """Stack the last-query attention pattern (log-probs) over the requested layers. Runs INSIDE a
    trace. Returns `[n_layers, n_heads, k_len]`."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():                                   # forward-only read (required on vLLM)
        for i in idx:
            # attention_interface_0.output = (attn_output, attn_weights); [1] is [B, heads, q, k].
            weights = blocks[i].attn.source.attention_interface_0.output[1]
            last = weights[0, :, -1, :]                     # last-query row -> [n_heads, k_len]
            rows.append(torch.log(last))                    # log so the oracle's softmax recovers A
    return torch.stack(rows, dim=0)                         # [n_layers, n_heads, k_len]


@cell("attention_pattern", family="gpt2", backend="hf")
def attention_pattern_gpt2_hf(be, model, prompts, *, layers="all"):
    return be.run(model, prompts, lambda: _attn_pattern(model.transformer.h, layers=layers))


@cell("attention_pattern", family="gpt2", backend="vllm_async")
def attention_pattern_gpt2_vllm(be, model, prompts, *, layers="all"):
    # Same explicit code as HF; the divergence is the backend's, not the cell's — vLLM has no
    # `attention_interface` op to read, so this raises and surfaces as ERROR.
    return be.run(model, prompts, lambda: _attn_pattern(model.transformer.h, layers=layers))

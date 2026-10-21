"""Activation harvesting: the residual stream at every prompt token, copied to host memory.

What each read denotes (nnsight 260c555b, transformers 5.12.1, vLLM 0.19.1):

  - HF: a block's output (or its `[0]`) is the residual stream after the block, `[batch, seq, d]`.
    A batched trace is left-padded to its longest prompt (`TransformersModel._configure_tokenizer`;
    docs/usage/invoke-and-batching.md, "Every invoke is padded to the whole batch's length"), and
    each prompt is tokenized on its own by the pipeline's `preprocess`, so prompt i's tokens are the
    last n_i positions of row i, n_i being its own token count. Pad positions hold computed values,
    so they are dropped, never exported.
  - vLLM: a served value is `[tokens, d]` for the request's own rows, with no batch axis and no
    padding (docs/models/vllm.md, "What your block sees on vLLM"). A fused-residual family's decoder
    layer returns `(hidden, residual)` and the stream after it is their sum
    (vllm/model_executor/models/qwen2.py); the default `residual` comes from the family profile.

Both cells copy the read to host memory inside the trace (`.cpu()`): vLLM returns a save on the
worker's device unless the block moves it, and a served value is the engine's live buffer, which a
later fused add+norm rewrites, so the copy is also what keeps the read intact (docs/models/vllm.md,
"Clone what you keep"). The output is `[tokens, d]` in prompt order on both engines, the layout of
the per-prompt reference. HF collects under `torch.no_grad()`, the documented collection form
(docs/patterns/probing.md): a trace otherwise runs the forward with autograd on.

`export=False` is the baseline: the same forward, returning each prompt's greedy next token.
"""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .registry import cell


@cell("activation_harvest", family="*", backend="hf")
def activation_harvest_hf(be, model, m, prompts, *, layer, export=True, residual="plain"):
    counts = [len(model.tokenizer(p)["input_ids"]) for p in prompts]

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        if not export:
            return be.last(m.head(model).output).argmax(-1).cpu()
        h = _resid(m.blocks(model)[layer].output, residual)             # [batch, seq, d]
        if h.shape[1] != max(counts):
            raise ValueError(f"batch is padded to {h.shape[1]} positions but the longest prompt "
                             f"tokenizes to {max(counts)}; valid tokens cannot be located")
        return torch.cat([row[row.shape[0] - n:] for row, n in zip(h, counts)]).cpu()

    with torch.no_grad():
        return be.run(model, prompts, build)


@cell("activation_harvest", family="*", backend="vllm_async")
def activation_harvest_vllm(be, model, m, prompts, *, layer, export=True, residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)

    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        if not export:
            return model.logits[-1:, :].argmax(-1).cpu()
        return _resid(m.blocks(model)[layer].output, residual).cpu()    # [tokens, d]

    return be.run(model, prompts, build)

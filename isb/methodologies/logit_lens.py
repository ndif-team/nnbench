"""Logit lens — fixed per-cell methodology (design.md §12.1).

Each cell passes `be.run` a `build()` closure that, *inside the trace*, reads GPT-2's own
residual stream and applies its own final-norm + unembed. The cells are explicit (they
name `model.transformer.h` etc.); they share `_lens_proxy` (bottom-up reuse, §12.1) and
differ only in their backend's `run` mechanics and the default `unembed` formulation —
which is the honest finding: logit-lens is near-portable, except vLLM's `lm_head.forward`
is guarded, so the portable form uses a weight matmul.

Variances (§12, the user's ask): `prompts` (list -> batched by `be.run`); `layers`
("all" | list[int], the observe set, abstract indices interpreted here against
`model.transformer.h`); `unembed` ("module" idiomatic | "weight" portable).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import cell


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _lens_proxy(blocks, norm, head, *, layers, unembed, last_fn):
    """Build the stacked logit-lens proxy. Runs INSIDE a trace (no trace-open here)."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():                              # forward-only; required on vLLM, harmless on HF
        for i in idx:
            normed = norm(_untuple(blocks[i].output))  # residual -> final norm
            logits = (
                F.linear(normed, head.weight)           # portable: bypass lm_head.forward guard
                if unembed == "weight"
                else head(normed)                       # idiomatic: model's own head
            )
            rows.append(last_fn(logits))                # last-token row
    return torch.stack(rows, dim=0)                     # [n_layers, ., vocab]


@cell("logit_lens", family="gpt2", backend="hf")
def logit_lens_gpt2_hf(be, model, prompts, *, layers="all", unembed="module"):
    return be.run(model, prompts, lambda: _lens_proxy(
        model.transformer.h, model.transformer.ln_f, model.lm_head,
        layers=layers, unembed=unembed, last_fn=be.last,
    ))


@cell("logit_lens", family="gpt2", backend="vllm_async")
def logit_lens_gpt2_vllm(be, model, prompts, *, layers="all", unembed="weight"):
    return be.run(model, prompts, lambda: _lens_proxy(
        model.transformer.h, model.transformer.ln_f, model.lm_head,
        layers=layers, unembed=unembed, last_fn=be.last,
    ))

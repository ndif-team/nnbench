"""Ablation (zero-knockout) — fixed per-cell methodology (design.md §12; registry entry "ablation").

Zero a component's output at one layer to measure its causal contribution, then read the model's
next-token distribution. Like steering this is a *write* methodology, but the write is a knockout
(`output = 0`) rather than an additive steer. The vLLM-safe form is whole-tuple **replacement**
(in-place raises on inference tensors, F-5).

Observable = the portable unembed of the final block's residual, last token (same readout as
steering/patching). `target="none"` skips the write -> the un-ablated baseline the effect-size guard
needs (a knockout that doesn't move the output makes a SUPPORTED verdict vacuous, cf. F-6).

Variances (params): `layer`; `target` ("mlp" | "attn" knock out that submodule, "none" = baseline);
`residual` ("plain" | "fused" — the dual-stream reconstruction, intervention-gaps Gap 1.2).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .logit_lens import _resid
from .registry import cell


def _ablate_and_read(target_module, blocks, norm, head, *, target, residual, last_fn):
    """Zero `target_module`'s output (unless target=="none"), then read final-block logits."""
    with torch.no_grad():
        if target != "none":
            out = target_module.output
            if isinstance(out, tuple):
                target_module.output = (torch.zeros_like(out[0]), *out[1:])  # zero stream 0, keep tail
            else:
                target_module.output = torch.zeros_like(out)                 # whole-tuple replace (vLLM-safe)
        normed = norm(_resid(blocks[-1].output, residual))
        logits = F.linear(normed, head.weight)
    return last_fn(logits)


def _target_module(block, target):
    if target in ("mlp", "none"):
        return block.mlp        # "none" never reads it; mlp is a harmless placeholder
    if target == "attn":
        return block.attn
    raise ValueError(f"unknown ablation target {target!r}")


@cell("ablation", family="gpt2", backend="hf")
def ablation_gpt2_hf(be, model, prompts, *, layer=6, target="mlp", residual="plain"):
    blk = model.transformer.h[layer]
    return be.run(model, prompts, lambda: _ablate_and_read(
        _target_module(blk, target), model.transformer.h, model.transformer.ln_f, model.lm_head,
        target=target, residual=residual, last_fn=be.last))


@cell("ablation", family="gpt2", backend="vllm_async")
def ablation_gpt2_vllm(be, model, prompts, *, layer=6, target="mlp", residual="plain"):
    blk = model.transformer.h[layer]
    return be.run(model, prompts, lambda: _ablate_and_read(
        _target_module(blk, target), model.transformer.h, model.transformer.ln_f, model.lm_head,
        target=target, residual=residual, last_fn=be.last))

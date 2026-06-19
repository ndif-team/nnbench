"""Ablation (zero-knockout) — fixed per-cell methodology (design.md §12; registry entry "ablation").

Zero a component's output at one layer to measure its causal contribution, then read the model's
next-token distribution. Like steering this is a *write* methodology, but the write is a knockout
(`output = 0`) rather than an additive steer. The vLLM-safe form is whole-tuple **replacement**
(in-place writes raise on inference tensors; replacement works).

Observable = the portable unembed of the final block's residual, last token (same readout as
steering/patching). `target="none"` skips the write -> the un-ablated baseline the effect-size guard
needs (a knockout that doesn't move the output makes a SUPPORTED verdict vacuous, cf. the
effect-size guard).

Variances (params): `layer`; `target` ("mlp" | "attn" knock out that submodule, "none" = baseline);
`residual` ("plain" | "fused" — the dual-stream reconstruction, where vLLM fused-residual blocks
return (hidden, residual) whose sum is the true stream).
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
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _ablate_and_read(
            _target_module(blk, target), model.transformer.h, model.transformer.ln_f, model.lm_head,
            target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)


@cell("ablation", family="gpt2", backend="vllm_async")
def ablation_gpt2_vllm(be, model, prompts, *, layer=6, target="mlp", residual="plain"):
    blk = model.transformer.h[layer]
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _ablate_and_read(
            _target_module(blk, target), model.transformer.h, model.transformer.ln_f, model.lm_head,
            target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)


# --- llama/Qwen family: blocks are model.model.layers; the attention submodule is `self_attn`
# (not GPT-2's `attn`), and the mlp is `mlp` (same name). _ablate_and_read is reused verbatim.
def _target_module_llama(block, target):
    if target in ("mlp", "none"):
        return block.mlp        # "none" never reads it; mlp is a harmless placeholder
    if target == "attn":
        return block.self_attn
    raise ValueError(f"unknown ablation target {target!r}")


@cell("ablation", family="llama", backend="hf")
def ablation_llama_hf(be, model, prompts, *, layer=6, target="mlp", residual="plain"):
    blk = model.model.layers[layer]
    def build():
        return _ablate_and_read(
            _target_module_llama(blk, target), model.model.layers, model.model.norm, model.lm_head,
            target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)


@cell("ablation", family="llama", backend="vllm_async")
def ablation_llama_vllm(be, model, prompts, *, layer=6, target="mlp", residual="plain"):
    blk = model.model.layers[layer]
    def build():
        return _ablate_and_read(
            _target_module_llama(blk, target), model.model.layers, model.model.norm, model.lm_head,
            target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)


# --- nemotron family (Nemotron-H / Nemotron 3): each block is a SINGLE op exposed as `block.mixer`
# (Mamba-2 / attention / MLP / MoE). There is no attn-vs-mlp split WITHIN a block, so ablation zeroes
# the whole layer's mixer output -> `hidden = residual + 0 = residual`, i.e. the layer becomes the
# identity. The GPT-2/Llama "which component" choice (mlp vs attn) becomes "which LAYER" (pick an index
# whose block_type is the one you want to knock out, e.g. a Mamba layer vs one of the few attention
# layers); the cell itself stays type-agnostic. `_ablate_and_read` is reused verbatim. (§12.7)
def _target_module_nemotron(block, target):
    if target in ("mixer", "none"):
        return block.mixer      # "none" never reads it; the single per-layer op for every block type
    raise ValueError(
        f"unknown ablation target {target!r} for nemotron: hybrid blocks expose one `.mixer`, so "
        f"pick the LAYER of the desired type rather than a within-block component (use 'mixer')"
    )


@cell("ablation", family="nemotron", backend="hf")
def ablation_nemotron_hf(be, model, prompts, *, layer=6, target="mixer", residual="plain"):
    blk = model.backbone.layers[layer]
    def build():
        return _ablate_and_read(
            _target_module_nemotron(blk, target), model.backbone.layers, model.backbone.norm_f,
            model.lm_head, target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)


@cell("ablation", family="nemotron", backend="vllm_async")
def ablation_nemotron_vllm(be, model, prompts, *, layer=6, target="mixer", residual="plain"):
    blk = model.backbone.layers[layer]
    def build():
        return _ablate_and_read(
            _target_module_nemotron(blk, target), model.backbone.layers, model.backbone.norm_f,
            model.lm_head, target=target, residual=residual, last_fn=be.last)
    return be.run(model, prompts, build)

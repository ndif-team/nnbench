"""Ablation (zero-knockout) — fixed per-cell methodology (design.md §12; registry entry "ablation").

Zero a component's output at one layer to measure its causal contribution, then read the model's
next-token distribution. Like steering this is a *write* methodology, but the write is a knockout
(`output = 0`) rather than an additive steer. The vLLM-safe form is whole-tuple **replacement**
(in-place writes raised on inference tensors before nnsight 0.8; replacement works).

Observable = the model's own next-token logits at the documented site, last token (same readout as
steering/patching: `readout.final_logits`). `target="none"` skips the write -> the un-ablated baseline the effect-size guard
needs (a knockout that doesn't move the output makes a SUPPORTED verdict vacuous, cf. the
effect-size guard).

Variances (params): `layer`; `target` ("mlp" | "attn" knock out that submodule, "none" = baseline).
The write zeroes a submodule's own output and the read is the model's logits, so no residual-stream
denotation enters the cell.
"""
from __future__ import annotations

import torch

from .readout import final_logits
from .registry import cell


def _ablate_and_read(target_module, model, head, *, target, site):
    """Zero `target_module`'s output (unless target=="none"), then read the next-token logits at
    `site` (`readout.final_logits`)."""
    with torch.no_grad():
        if target != "none":
            out = target_module.output
            if isinstance(out, tuple):
                target_module.output = (torch.zeros_like(out[0]), *out[1:])  # zero stream 0, keep tail
            else:
                target_module.output = torch.zeros_like(out)                 # whole-tuple replace (vLLM-safe)
        return final_logits(model, head, site)


def _target_module(block, target):
    if target in ("mlp", "none"):
        return block.mlp        # "none" never reads it; mlp is a harmless placeholder
    if target == "attn":
        return block.attn
    raise ValueError(f"unknown ablation target {target!r}")


@cell("ablation", family="*", backend="hf")
def ablation_hf(be, model, m, prompts, *, layer=6, target="mlp"):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)  # "none" never reads it
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="head")
    return be.run(model, prompts, build)


@cell("ablation", family="*", backend="vllm_async")
def ablation_vllm(be, model, m, prompts, *, layer=6, target="mlp"):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)  # "none" never reads it
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="engine")
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
def ablation_nemotron_hf(be, model, prompts, *, layer=6, target="mixer"):
    blk = model.model.layers[layer]
    def build():
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="head")
    return be.run(model, prompts, build)


@cell("ablation", family="nemotron", backend="vllm_async")
def ablation_nemotron_vllm(be, model, prompts, *, layer=6, target="mixer"):
    # Same tree as HF (model.model.layers, block.mixer); the read is the engine's logits.
    blk = model.model.layers[layer]
    def build():
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="engine")
    return be.run(model, prompts, build)

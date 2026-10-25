"""Ablation (zero knockout): replace a block component's output with zeros and read the model's
next-token logits. `target="none"` skips the write (the baseline). The nemotron family's blocks are
single ops exposed as `block.mixer`, so there the choice of component is the choice of layer."""
from __future__ import annotations

import torch

from .readout import final_logits
from .registry import cell


def _ablate_and_read(target_module, model, head, *, target, site):
    """Zero `target_module`'s output (unless target == "none"), then read the next-token logits."""
    with torch.no_grad():
        if target != "none":
            out = target_module.output
            if isinstance(out, tuple):
                target_module.output = (torch.zeros_like(out[0]), *out[1:])
            else:
                target_module.output = torch.zeros_like(out)
        return final_logits(model, head, site)


@cell("ablation", family="*", backend="hf")
def ablation_hf(be, model, m, units, *, layer=6, target="mlp"):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)  # "none" never reads it

    def build(rows):
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="head")

    return be.run(model, units, build)


@cell("ablation", family="*", backend="vllm_async")
def ablation_vllm(be, model, m, units, *, layer=6, target="mlp"):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)

    def build(rows):
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="engine")

    return be.run(model, units, build)


def _target_module_nemotron(block, target):
    if target in ("mixer", "none"):
        return block.mixer
    raise ValueError(f"unknown ablation target {target!r} for nemotron: hybrid blocks expose one "
                     f"`.mixer`, so pick the layer of the desired type (use 'mixer')")


@cell("ablation", family="nemotron", backend="hf")
def ablation_nemotron_hf(be, model, units, *, layer=6, target="mixer"):
    blk = model.model.layers[layer]

    def build(rows):
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="head")

    return be.run(model, units, build)


@cell("ablation", family="nemotron", backend="vllm_async")
def ablation_nemotron_vllm(be, model, units, *, layer=6, target="mixer"):
    blk = model.model.layers[layer]

    def build(rows):
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="engine")

    return be.run(model, units, build)

"""Ablation (zero knockout): replace a block component's output with zeros and return the model's
greedy next token; a check call returns the next-token logits instead. `target="none"` skips the
write (the baseline). The nemotron family's blocks are single ops exposed as `block.mixer`, so
there the choice of component is the choice of layer."""
from __future__ import annotations

import torch

from ..oracle.checks import timed_check_of, tokens_match
from .readout import final_logits, next_token
from .registry import cell

timed_check_of("ablation")(tokens_match)


def _ablate_and_read(target_module, model, head, *, target, site, read=final_logits):
    """Zero `target_module`'s output (unless target == "none"), then return `read(model, head,
    site)`: the next token (`next_token`) or the next-token logits (`final_logits`)."""
    with torch.no_grad():
        if target != "none":
            out = target_module.output
            if isinstance(out, tuple):
                target_module.output = (torch.zeros_like(out[0]), *out[1:])
            else:
                target_module.output = torch.zeros_like(out)
        return read(model, head, site)


@cell("ablation", family="*", backend="hf")
def ablation_hf(be, model, m, units, *, layer=6, target="mlp", check=False):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)  # "none" never reads it

    def build(rows):
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="head",
                                read=final_logits if check else next_token)

    return be.run(model, units, build, keep_dtype=not check)


@cell("ablation", family="*", backend="vllm_async")
def ablation_vllm(be, model, m, units, *, layer=6, target="mlp", check=False):
    blk = m.blocks(model)[layer]
    tgt = m.submodule(blk, "mlp" if target == "none" else target)

    def build(rows):
        return _ablate_and_read(tgt, model, m.head(model), target=target, site="engine",
                                read=final_logits if check else next_token)

    return be.run(model, units, build, keep_dtype=not check)


def _target_module_nemotron(block, target):
    if target in ("mixer", "none"):
        return block.mixer
    raise ValueError(f"unknown ablation target {target!r} for nemotron: hybrid blocks expose one "
                     f"`.mixer`, so pick the layer of the desired type (use 'mixer')")


@cell("ablation", family="nemotron", backend="hf")
def ablation_nemotron_hf(be, model, units, *, layer=6, target="mixer", check=False):
    blk = model.model.layers[layer]

    def build(rows):
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="head",
                                read=final_logits if check else next_token)

    return be.run(model, units, build, keep_dtype=not check)


@cell("ablation", family="nemotron", backend="vllm_async")
def ablation_nemotron_vllm(be, model, units, *, layer=6, target="mixer", check=False):
    blk = model.model.layers[layer]

    def build(rows):
        return _ablate_and_read(_target_module_nemotron(blk, target), model, model.lm_head,
                                target=target, site="engine",
                                read=final_logits if check else next_token)

    return be.run(model, units, build, keep_dtype=not check)

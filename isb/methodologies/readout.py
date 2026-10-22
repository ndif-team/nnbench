"""Documented read sites shared by the write methodologies, and the steering direction.

nnsight 0.8.1 documents where a cell reads the model's next-token logits and how it brings in a
vector computed outside the trace (docs at the pinned tag):

- HF: the model's own head, `model.lm_head.output[:, -1, :]` (docs/patterns/steering.md).
- vLLM: the engine site `model.logits`, `[1, vocab]` for the step (docs/models/vllm.md, "What your
  block sees on vLLM"). It is whole under tensor parallelism, where a parameter is only this rank's
  slice (docs/models/vllm-parallelism.md), so the logits are never rebuilt from `lm_head.weight`.
- Under transformers tensor parallelism (docs/models/tensor-parallel.md) the same holds on HF:
  `lm_head.output` and ad hoc module calls arrive whole, a parameter is a `DTensor` slice.
- A tensor the block references from outside travels with the block to every rank; move it onto
  the served value inside the block (`v.to(h.device, h.dtype)`), which also covers an HF model whose
  layers `device_map` spread over several GPUs.

The steering direction is the target token's unembedding row, built before the trace: from the
loaded head on HF, and from the checkpoint on vLLM, whose client holds no weights.
"""
from __future__ import annotations

import functools
import json
import os

import torch
import torch.nn.functional as F

SITES = ("head", "engine")


def final_logits(model, head, site):
    """The last token's next-token logits at the documented read site. Runs INSIDE a trace.
    "head": the model's own head output (HF), `[B, vocab]`. "engine": vLLM's `model.logits`,
    `[1, vocab]`."""
    if site == "engine":
        return model.logits[-1:, :]
    if site == "head":
        return head.output[:, -1, :]
    raise ValueError(f"unknown read site {site!r} (expected one of {SITES})")


def head_direction(model, head, token_id):
    """Unit unembedding row of `token_id` from a loaded head (HF), fp32, on the head's device.
    Built OUTSIDE the trace; the block moves it onto the stream it writes. Under transformers
    tensor parallelism the head's weight is a `DTensor` holding this rank's vocabulary slice
    (nnsight docs/models/tensor-parallel.md, "Parameters are DTensors"), so the row is read from
    the checkpoint instead, as on vLLM."""
    weight = head.weight
    if hasattr(weight, "placements"):                    # a DTensor: sharded, not gathered
        return checkpoint_direction(checkpoint_of(model), token_id)
    with torch.no_grad():
        return F.normalize(weight[token_id].detach().float(), dim=0)


# Checkpoint keys of the unembedding: the head's own, else the tied input embedding.
_UNEMBED_KEYS = ("lm_head.weight", "model.embed_tokens.weight", "transformer.wte.weight",
                 "wte.weight", "backbone.embeddings.weight")


def checkpoint_of(model):
    """The checkpoint a model was loaded from: nnsight's `repo_id` where the model has one, else
    its tokenizer's `name_or_path` (nnsight's VLLM loads both from the same repo)."""
    return getattr(model, "repo_id", None) or model.tokenizer.name_or_path


def _checkpoint_file(repo, filename):
    if os.path.isdir(repo):                              # a local checkpoint directory
        path = os.path.join(repo, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


@functools.lru_cache(maxsize=None)
def checkpoint_direction(repo, token_id):
    """Unit unembedding row of `token_id` read from the cached checkpoint (vLLM, whose client holds
    no weights), fp32 on the CPU. Cached per (repo, token): the first call reads one row from disk."""
    from safetensors import safe_open

    try:
        index = json.load(open(_checkpoint_file(repo, "model.safetensors.index.json")))["weight_map"]
    except Exception:                                    # a single-file checkpoint
        index = None
    names = list(index) if index is not None else None
    for key in _UNEMBED_KEYS:
        if names is not None and key not in names:
            continue
        path = _checkpoint_file(repo, index[key] if index is not None else "model.safetensors")
        with safe_open(path, framework="pt") as f:
            if key not in f.keys():
                continue
            row = f.get_slice(key)[token_id].float()
        return F.normalize(row, dim=0)
    raise KeyError(f"{repo}: no unembedding weight among {_UNEMBED_KEYS}")

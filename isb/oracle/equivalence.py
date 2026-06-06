"""Equivalence oracle (design.md §8.3, §11.8).

Compares a backend's result against the HF-eager reference. This is what distinguishes
`SUPPORTED` from `SILENTLY_WRONG` (§8.1) — the cell only a numerical check can detect.

For logit-lens the output is per-layer logits over the vocab; we use **top-1 token
agreement per layer** (robust to vLLM's different kernels/dtype) plus a max-abs-diff
diagnostic. Other motifs can supply their own comparator later.
"""
from __future__ import annotations


def compare(ref, got) -> dict:
    import torch

    if ref is None or got is None:
        return {"shape_match": False, "top1_agree": 0.0, "max_abs": float("inf")}
    r = ref.float()
    g = got.float()
    shape_match = tuple(r.shape) == tuple(g.shape)
    if not shape_match:
        r2, g2 = r.squeeze(), g.squeeze()
        if tuple(r2.shape) == tuple(g2.shape):
            r, g, shape_match = r2, g2, True
    rt = r.argmax(-1).flatten()
    gt = g.argmax(-1).flatten()
    top1 = (rt == gt).float().mean().item() if rt.shape == gt.shape else 0.0
    max_abs = (r - g).abs().max().item() if shape_match else float("inf")
    return {"shape_match": shape_match, "top1_agree": top1, "max_abs": max_abs}


def is_equivalent(metrics: dict, top1_thresh: float = 0.8) -> bool:
    return metrics["shape_match"] and metrics["top1_agree"] >= top1_thresh

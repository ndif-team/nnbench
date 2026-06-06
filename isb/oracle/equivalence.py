"""Equivalence oracle (design.md §8.3, §11.8).

Compares a backend's result against the HF-eager reference. This is what distinguishes
`SUPPORTED` from `SILENTLY_WRONG` (§8.1) — the cell only a numerical check can detect.

For a logits-producing motif (logit lens), the output IS a per-row distribution. We gate
on TWO criteria so a top-1-preserving divergence cannot pass as SUPPORTED:
  - **top-1 token agreement** per row (robust to kernel/dtype), and
  - **softmax total-variation distance** (the distributional difference).
A uniform additive logit shift leaves both top-1 and softmax unchanged — correctly judged
equivalent, since it does not change the lens's prediction. `max_abs` on raw logits is a
diagnostic only (logit scale is not itself a correctness signal), never a gate.
"""
from __future__ import annotations


def compare(ref, got) -> dict:
    import torch

    miss = {
        "shape_match": False, "top1_agree": 0.0, "tv": 1.0,
        "max_abs": float("inf"), "has_ref": ref is not None,
    }
    if ref is None or got is None:
        return miss
    r = ref.float()
    g = got.float()
    # vLLM's ParallelLMHead pads the vocab (50257->50304); align the real vocab so
    # padding positions don't skew argmax / distance.
    if r.shape[-1] != g.shape[-1]:
        m = min(r.shape[-1], g.shape[-1])
        r, g = r[..., :m], g[..., :m]
    # rescue [.,1,V] vs [.,V] alignment ONLY when squeezing makes shapes equal.
    if r.shape != g.shape and r.squeeze().shape == g.squeeze().shape:
        r, g = r.squeeze(), g.squeeze()
    if tuple(r.shape) != tuple(g.shape):
        return {**miss, "has_ref": True}

    rt = r.argmax(-1).flatten()
    gt = g.argmax(-1).flatten()
    top1 = (rt == gt).float().mean().item()
    pr = r.softmax(-1)
    pg = g.softmax(-1)
    tv = (0.5 * (pr - pg).abs().sum(-1)).mean().item()  # mean total variation in [0,1]
    max_abs = (r - g).abs().max().item()
    return {"shape_match": True, "top1_agree": top1, "tv": tv, "max_abs": max_abs, "has_ref": True}


def is_equivalent(metrics: dict, top1_thresh: float = 0.9, tv_tol: float = 0.05) -> bool:
    return (
        metrics["shape_match"]
        and metrics["top1_agree"] >= top1_thresh
        and metrics["tv"] <= tv_tol
    )

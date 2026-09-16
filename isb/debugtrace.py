"""The release/debug run split.

A normal run file is the RELEASE artifact: the lean cells users actually run, in the regime
perf is measured in. Verdicts and perf numbers come only from it. `--debug` additionally
writes a companion file `debug/<name>-debug.pt` holding an instrumented forward over the
first few regime inputs: token ids, the residual stream at every layer (readout position),
and the final logits.

The companion never feeds a verdict: extra saves change what the engine executes, so an
instrumented run is a different program than the one being certified. Its job is to EXPLAIN a
divergence the release file already shows — `scripts/debug_compare.py` diffs two companions
layer by layer and reports where the engines start to drift.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .profiles import _resid

N_DEBUG_PROMPTS = 2
DRIFT_FLOOR = 1e-3          # relative difference below this reads as kernel noise, not drift


def debug_prompts(spec, k: int = N_DEBUG_PROMPTS) -> list[str]:
    """The companion's inputs: the first k prompts of the spec's first execution regime. Pair and
    labeled-pair units contribute their first (clean) prompt."""
    return [u[0] if isinstance(u, tuple) else u for u in spec.regimes[0].prompts[:k]]


def collect_debug(be, model, m, prompts, residual: str) -> tuple[dict, dict]:
    """Instrumented per-prompt traces -> (outputs, extra_provenance). Outputs: for prompt i,
    ("debug_resid", i) = [n_layers, d] residual at the last position per layer (fp32), and
    ("debug_logits", i) = [vocab] final logits through the portable unembed."""
    blocks, ln_f, head = m.blocks(model), m.norm(model), m.head(model)

    def resid_stack():  # named (not a lambda) so nnsight can source-serialize it to the worker
        with torch.no_grad():
            rows = []
            for blk in blocks:
                h = _resid(blk.output, residual)
                rows.append(h.reshape(-1, h.shape[-1])[-1, :].float())
            return torch.stack(rows)

    def final_logits():  # named for the same serialization reason
        with torch.no_grad():
            normed = ln_f(_resid(blocks[-1].output, residual))
            logits = F.linear(normed.float(), head.weight.float())
            return logits.reshape(-1, logits.shape[-1])[-1, :]

    outputs = {}
    for i, p in enumerate(prompts):
        outputs[("debug_resid", i)] = be.run(model, [p], resid_stack)
        outputs[("debug_logits", i)] = be.run(model, [p], final_logits)
    extra = {"debug": {
        "prompts": list(prompts),
        "token_ids": [model.tokenizer(p)["input_ids"] for p in prompts],
        "residual": residual,
    }}
    return outputs, extra


def compare_debug(cand: dict, ref: dict) -> list[dict]:
    """Layer-by-layer diff of two companion files' outputs. One row per prompt:
    per-layer relative difference of the residual, the first layer past DRIFT_FLOOR, and
    final-logit agreement (top-1 match + total variation on the common vocab)."""
    rows = []
    for key in sorted(k for k in cand if k[0] == "debug_resid"):
        i = key[1]
        c, r = cand[key].float(), ref[key].float()
        if c.shape != r.shape:
            rows.append({"prompt": i, "error": f"residual shape {tuple(c.shape)} != {tuple(r.shape)}"})
            continue
        rel = ((c - r).norm(dim=-1) / (r.norm(dim=-1) + 1e-12)).tolist()
        first = next((L for L, v in enumerate(rel) if v > DRIFT_FLOOR), None)
        cl, rl = cand[("debug_logits", i)].float(), ref[("debug_logits", i)].float()
        v = min(cl.shape[-1], rl.shape[-1])          # vLLM pads the vocab; extra columns are dead
        pc, pr = F.softmax(cl[:v], -1), F.softmax(rl[:v], -1)
        rows.append({
            "prompt": i,
            "per_layer_rel": rel,
            "first_drift_layer": first,
            "top1_match": bool(pc.argmax() == pr.argmax()),
            "logit_tv": float(0.5 * (pc - pr).abs().sum()),
        })
    return rows

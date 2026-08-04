"""Reproduce the Jacobian-lens `lens-eval-*` datasets (anthropics/jacobian-lens) as one task.

All six sets (`data/jlens/`) are this ONE function; per the upstream README they differ
only in the readout-position rule, a task-level knob:

  - association / multihop / multilingual / order-ops / typo: readout at the final prompt
    token (the token immediately preceding the would-be target)      -> position="last"
  - poetry: readout at the last newline (end of line 1 of the couplet) -> position="last_newline"

`transport=None` reads the plain logit lens at that position; `transport=[J_1..J_L]` (the
fitted per-layer Jacobian maps, e.g. loaded from the upstream `JacobianLens` checkpoint)
reads the J-lens: `unembed(norm(J_l @ h))`, the logit lens with one extra matmul before
the final norm. Scoring is upstream pass@k: mean over items of the fraction of
`intermediates` whose min-over-layers rank <= k at the readout position.

The task is free-form code (no trace schema); portability comes from `be` (backend
interface) and `m` (ModelProfile): the same function runs hf / vllm_async / vllm_serve on
any profiled family.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from ..profiles import _resid


def load_items(path: str) -> list:
    with open(path) as f:
        return json.load(f)["items"]


def locate(tokenizer, prompt: str, rule: str) -> int:
    """Readout token index into the prompt (input-space, per item). Indices are computed on
    `tokenizer.encode(prompt)` WITH special tokens, matching the sequence the model actually
    runs (e.g. a llama BOS shifts absolute indices; `last` is unaffected, `last_newline` is)."""
    if rule == "last":
        return -1
    if rule == "last_newline":
        ids = tokenizer.encode(prompt)
        nl = [i for i, tid in enumerate(ids) if "\n" in tokenizer.decode([tid])]
        if not nl:
            raise ValueError("last_newline readout on a prompt with no newline token")
        return nl[-1]
    raise ValueError(f"unknown position rule {rule!r}")


def first_token_id(tokenizer, word: str) -> int:
    """The vocabulary token tracked for `word`: leading-space form, first fragment (the
    single-token convention; a multi-fragment word is tracked by its first fragment)."""
    return tokenizer.encode(" " + word.strip(), add_special_tokens=False)[0]


def lens_eval(be, model, m, items, *, band="all", position="last", unembed="weight",
              residual="plain", transport=None):
    """Per item: the [n_band_layers, vocab] lens logits at the readout position."""
    def run_one(item):
        prompt = item["prompt"]
        pos = locate(model.tokenizer, prompt, position)

        def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
            blocks, norm, head = m.blocks(model), m.norm(model), m.head(model)
            idx = range(len(blocks)) if band == "all" else band
            rows = []
            with torch.no_grad():
                for i in idx:
                    # position slice before any projection: the seq dim is -2 on BOTH backends
                    # (hf [B, S, D], vllm flat [S, D]), so one indexing form covers both layouts
                    h = _resid(blocks[i].output, residual)[..., pos, :]
                    if transport is not None:
                        h = h @ transport[i].T          # J-lens transport, BEFORE the final norm
                    normed = norm(h)
                    rows.append(F.linear(normed, head.weight) if unembed == "weight"
                                else head(normed))
            return torch.stack(rows)                    # [n_band_layers, ., vocab]
        return be.run(model, [prompt], build)

    return [run_one(it) for it in items]


def pass_at_k(outs, items, tokenizer, k: int = 10) -> float:
    """Upstream metric (data/jlens/README-upstream.md): mean over items of the fraction of
    `intermediates` whose min-over-layers rank <= k. `outs[i]` is lens_eval's per-item output."""
    fracs = []
    for out, item in zip(outs, items):
        logits = out.reshape(out.shape[0], -1, out.shape[-1])[:, -1, :]   # [L, vocab], B=1
        # rank of every vocab id per layer (0-based): double argsort of descending logits
        ranks = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        words = item["intermediates"]
        hits = sum(1 for w in words if int(ranks[:, first_token_id(tokenizer, w)].min()) < k)
        fracs.append(hits / len(words))
    return sum(fracs) / len(fracs)


# position rule per upstream dataset: single copy in the data registry (isb/data.py), where it is
# also injected automatically as a dataset knob for spec runs
from ..data import POSITION_RULES  # noqa: E402,F401

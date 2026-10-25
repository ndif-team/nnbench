"""Attribution patching (Nanda, 2023): the first-order estimate of activation patching over every
layer from one clean forward and one corrupt forward + backward per pair,
`((a_clean - a_corrupt) * dM/da_corrupt).sum()`, with the metric `logit[clean answer] -
logit[corrupt answer]` at the last token. A unit is a labeled pair (clean, corrupt, (correct,
incorrect)). Needs autograd: HF runs it, vLLM's inference-mode activations refuse it (the recorded
verdict). `grad=False` is the forward-only metric on the corrupt prompts (the baseline), `[B, 1]`."""
from __future__ import annotations

import torch

from ..profiles import _resid
from .observations import record_resolved
from .readout import final_logits
from .registry import cell


def _answer_ids(model, units):
    """(clean answer id, corrupt answer id) per unit: the first token of each answer, with no
    special prefix (a Llama-3 tokenizer prepends BOS by default)."""
    def first_id(text):
        return model.tokenizer.encode(text, add_special_tokens=False)[0]
    return [(first_id(unit[2][0]), first_id(unit[2][1])) for unit in units]


def _attribution_cell(be, model, m, units, *, residual, grad, site):
    clean = [unit[0] for unit in units]
    corrupt = [unit[1] for unit in units]
    ids = _answer_ids(model, units)
    blocks, head = m.blocks(model), m.head(model)

    if not grad:   # baseline: the forward-only metric of every corrupt run, one row per unit
        def build(rows):
            last = final_logits(model, head, site)                        # [B_trace, vocab]
            clean_ids = torch.tensor([ids[r][0] for r in rows], device=last.device)
            corrupt_ids = torch.tensor([ids[r][1] for r in rows], device=last.device)
            index = torch.arange(len(rows), device=last.device)
            return (last[index, clean_ids] - last[index, corrupt_ids]).unsqueeze(-1)   # [B_trace, 1]

        return be.run(model, corrupt, build)

    def acts_of(mdl):
        return [_resid(blk.output, residual) for blk in m.blocks(mdl)]

    def metric_of(mdl, i):
        last = final_logits(mdl, m.head(mdl), site)[0]
        return last[ids[i][0]] - last[ids[i][1]]

    return be.attribute(model, clean, corrupt, acts_of=acts_of, metric_of=metric_of, n=len(blocks))


@cell("attribution_patching", family="*", backend="hf")
def attribution_patching_hf(be, model, m, units, *, residual="plain", grad=True):
    return _attribution_cell(be, model, m, units, residual=residual, grad=grad, site="head")


@cell("attribution_patching", family="*", backend="vllm_async", realizations=[{"residual": "plain"}])
def attribution_patching_vllm(be, model, m, units, *, residual=None, grad=True):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    return _attribution_cell(be, model, m, units, residual=residual, grad=grad, site="engine")

"""Attribution patching (Nanda, 2023) — fixed per-cell methodology (design.md §12).

A first-order linear approximation of activation patching: where full patching costs one forward per
component, attribution patching gets a saliency over every layer from one clean forward and one
corrupt forward+backward. For a residual activation `a` and metric `M`:

    M(a_clean) - M(a_corrupt) ≈ (a_clean - a_corrupt) · ∂M/∂a │ a = a_corrupt

so the per-layer attribution is `((a_clean - a_corrupt) * grad_corrupt).sum()`. Metric here is the
logit difference `logit[clean_answer] - logit[corrupt_answer]` on the corrupt run (portable unembed,
so the weight matmul sidesteps vLLM's guarded `lm_head.forward`); a high positive score at layer L means "patching L's clean
residual would most raise the clean-vs-corrupt logit gap". The unit is a LABELED pair
(clean, corrupted, (correct, incorrect)) — the per-item answers come from the data source
(mib/ioi_labeled stores the MIB snapshot's indirect_object/subject strings), so the verdict
aggregates the [n_layers] attribution over a real pair set instead of one hand-written pair.

**Backend frontier — this cell exercises the `grad` primitive.** It needs autograd (a backward pass).
HF supports it -> SUPPORTED. vLLM runs in inference mode; its activations are inference tensors, so
`requires_grad_` / backward raise -> ERROR. That HF-vs-vLLM split (a whole gradient-based class is
HF-only) is the finding. The clean/corrupt prompts are a length-matched minimal pair so the residual
shapes align and the subtraction is position-aligned.

Variances (params): `residual` ("plain" | "fused", the same reconstruction as logit-lens, where
vLLM fused-residual blocks return (hidden, residual) whose sum is the true stream);
`grad` (True = the attribution; False = the forward-only baseline, the overhead denominator — it
reads the metric with no backward and so runs on both backends).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .registry import cell


def _metric(blocks, norm, head, residual, clean_id, corrupt_id):
    """logit[clean] - logit[corrupt] at the last token, via the portable unembed. Runs in a trace.
    Layout-agnostic last-position slice: HF is [1, seq, vocab], vLLM flattens to [tokens, vocab]."""
    normed = norm(_resid(blocks[-1].output, residual))
    logits = F.linear(normed, head.weight)
    last = logits.reshape(-1, logits.shape[-1])[-1]
    return last[clean_id] - last[corrupt_id]


def _attribution_cell(be, model, m, prompts, *, residual, grad):
    clean, corrupt, answers = prompts
    clean_id = model.tokenizer.encode(answers[0])[0]     # correct answer for the clean prompt
    corrupt_id = model.tokenizer.encode(answers[1])[0]   # the competing (incorrect) answer
    blocks, ln_f, head = m.blocks(model), m.norm(model), m.head(model)

    if not grad:   # baseline: forward-only metric on the corrupt run (no backward), both backends
        def build():  # named so nnsight can source-serialize it to the vLLM worker
            return _metric(blocks, ln_f, head, residual, clean_id, corrupt_id)
        return be.run(model, [corrupt], build)

    def acts_of(mdl):  # named (not a lambda) so nnsight can source-serialize to the vLLM worker
        return [_resid(blk.output, residual) for blk in m.blocks(mdl)]

    def metric_of(mdl):
        return _metric(m.blocks(mdl), m.norm(mdl), m.head(mdl), residual, clean_id, corrupt_id)

    return be.attribute(
        model, clean, corrupt, acts_of=acts_of, metric_of=metric_of,
        n=len(blocks),
    )


@cell("attribution_patching", family="*", backend="hf")
def attribution_patching_hf(be, model, m, prompts, *, residual="plain", grad=True):
    return _attribution_cell(be, model, m, prompts, residual=residual, grad=grad)


@cell("attribution_patching", family="*", backend="vllm_async")
def attribution_patching_vllm(be, model, m, prompts, *, residual="plain", grad=True):
    # Same explicit code as HF; the divergence is the backend's: vLLM has no autograd, so the
    # backward in `be.attribute` raises and surfaces as ERROR.
    return _attribution_cell(be, model, m, prompts, residual=residual, grad=grad)

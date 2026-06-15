"""Attribution patching (Nanda, 2023) — fixed per-cell methodology (design.md §12).

A first-order linear approximation of activation patching: where full patching costs one forward per
component, attribution patching gets a saliency over every layer from one clean forward and one
corrupt forward+backward. For a residual activation `a` and metric `M`:

    M(a_clean) - M(a_corrupt) ≈ (a_clean - a_corrupt) · ∂M/∂a │ a = a_corrupt

so the per-layer attribution is `((a_clean - a_corrupt) * grad_corrupt).sum()`. Metric here is the
logit difference `logit[clean_answer] - logit[corrupt_answer]` on the corrupt run (portable unembed,
so the weight matmul sidesteps vLLM's guarded `lm_head.forward`); a high positive score at layer L means "patching L's clean
residual would most raise the clean-vs-corrupt logit gap".

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

from .logit_lens import _resid
from .registry import cell

CLEAN_ANSWER = " Paris"      # answer for "...France..."
CORRUPT_ANSWER = " Moscow"   # answer for "...Russia..."


def _metric(blocks, norm, head, residual, clean_id, corrupt_id):
    """logit[clean] - logit[corrupt] at the last token, via the portable unembed. Runs in a trace."""
    normed = norm(_resid(blocks[-1].output, residual))
    logits = F.linear(normed, head.weight)[:, -1, :]
    return logits[:, clean_id] - logits[:, corrupt_id]


def _attribution_cell(be, model, prompts, *, residual, grad):
    clean, corrupt = prompts
    clean_id = model.tokenizer.encode(CLEAN_ANSWER)[0]
    corrupt_id = model.tokenizer.encode(CORRUPT_ANSWER)[0]

    if not grad:   # baseline: forward-only metric on the corrupt run (no backward) — both backends
        return be.run(model, [corrupt], lambda: _metric(
            model.transformer.h, model.transformer.ln_f, model.lm_head, residual, clean_id, corrupt_id))

    return be.attribute(
        model, clean, corrupt,
        acts_of=lambda m: [_resid(blk.output, residual) for blk in m.transformer.h],
        metric_of=lambda m: _metric(
            m.transformer.h, m.transformer.ln_f, m.lm_head, residual, clean_id, corrupt_id),
        n=len(model.transformer.h),
    )


@cell("attribution_patching", family="gpt2", backend="hf")
def attribution_patching_gpt2_hf(be, model, prompts, *, residual="plain", grad=True):
    return _attribution_cell(be, model, prompts, residual=residual, grad=grad)


@cell("attribution_patching", family="gpt2", backend="vllm_async")
def attribution_patching_gpt2_vllm(be, model, prompts, *, residual="plain", grad=True):
    # Same explicit code as HF; the divergence is the backend's — vLLM has no autograd, so the
    # backward in `be.attribute` raises and surfaces as ERROR.
    return _attribution_cell(be, model, prompts, residual=residual, grad=grad)

"""Logit lens — fixed per-cell methodology (design.md §12.1).

Each cell passes `be.run` a `build()` closure that, *inside the trace*, reads GPT-2's own
residual stream and applies its own final-norm + unembed. The cells are explicit (they
name `model.transformer.h` etc.); they share `_lens_proxy` (bottom-up reuse, §12.1) and
differ only in their backend's `run` mechanics and the default `unembed` formulation —
which is the honest finding: logit-lens is near-portable, except vLLM's `lm_head.forward`
is guarded, so the portable form uses a weight matmul.

Variances (§12, the user's ask): `prompts` (list -> batched by `be.run`); `layers`
("all" | list[int], the observe set, abstract indices interpreted here against
`model.transformer.h`); `unembed` ("module" idiomatic | "weight" portable).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .registry import cell


def _untuple(x):
    # PP cross-stage outputs are LazyRemoteTensor, which is NOT a tuple instance even when it wraps a
    # (hidden, residual) tuple — so `isinstance(x, tuple)` is unreliable under pipeline parallelism and
    # would pass the lazy-wrapped tuple straight into RMSNorm (-> empty_like(tuple) TypeError on Qwen/
    # Llama, whose blocks return tuples). Probe tensor-ness first and index [0] on everything else:
    # real tuples index cleanly, and LazyRemoteTensor[0] returns a deferred child that pulls element 0.
    # Matches the codebase convention (nnsight tests/vllm/pp/manual run_equivalence_matrix._hidden).
    import torch
    return x if isinstance(x, torch.Tensor) else x[0]


def _resid(out, how):
    """The residual stream at a block's output boundary.

    `plain`: the block output (or `[0]` of its tuple) — correct for GPT-2 and for any HF block,
    which already carries the full residual stream.
    `fused`: `hidden + residual`. This is the **documented vLLM dual-residual-stream issue** (the vLLM dual
    residual stream): vLLM's Llama/RMSNorm decoder layers return `(hidden, residual)` and the true
    residual stream is their SUM — vLLM computes exactly `hidden + residual` for its own aux hidden
    states (vllm .../models/llama.py:425; VLLM_GUIDE "Logit Lens" prescribes `out[0] + out[1]`).
    Reading only `[0]` (as `plain` does) drops the accumulated residual -> silently wrong logits.
    Falls back to `plain` when the output is not a `(hidden, residual)` container, so it is safe on HF.
    The second branch handles **pipeline parallelism**: a PP cross-stage output is a `LazyRemoteTensor`
    wrapping `(hidden, residual)` — NOT a tuple instance — so `isinstance(out, tuple)` alone would skip
    the fused branch and silently drop the residual on the PP candidate (the same reason `_untuple`
    probes tensor-ness). `fused` is only ever requested by vLLM dual-residual cells, where a non-tensor
    output always carries `(hidden, residual)`.
    """
    if how == "fused":
        if isinstance(out, tuple) and len(out) >= 2:        # real tuple (HF / single-GPU vLLM)
            return out[0] + out[1]
        if not isinstance(out, (torch.Tensor, tuple)):      # PP LazyRemoteTensor wrapping (hidden, residual)
            return out[0] + out[1]
    return _untuple(out)


def _lens_proxy(blocks, norm, head, *, layers, unembed, last_fn, residual="plain"):
    """Build the stacked logit-lens proxy. Runs INSIDE a trace (no trace-open here)."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():                                   # forward-only; required on vLLM, harmless on HF
        for i in idx:
            normed = norm(_resid(blocks[i].output, residual))  # residual stream -> final norm
            logits = (
                F.linear(normed, head.weight)           # portable: bypass lm_head.forward guard
                if unembed == "weight"
                else head(normed)                       # idiomatic: model's own head
            )
            rows.append(last_fn(logits))                # last-token row
    return torch.stack(rows, dim=0)                     # [n_layers, ., vocab]


@cell("logit_lens", family="gpt2", backend="hf")
def logit_lens_gpt2_hf(be, model, prompts, *, layers="all", unembed="module"):
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            model.transformer.h, model.transformer.ln_f, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last,
        )
    return be.run(model, prompts, build)


@cell("logit_lens", family="gpt2", backend="vllm_async")
def logit_lens_gpt2_vllm(be, model, prompts, *, layers="all", unembed="weight"):
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            model.transformer.h, model.transformer.ln_f, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last,
        )
    return be.run(model, prompts, build)


# --- Llama family: same methodology, the model's OWN module names (§12.1 explicit-per-family).
# `transformer.h`/`ln_f` (GPT-2) become `model.layers`/`model.norm` (Llama); the final norm is
# RMSNorm not LayerNorm. _lens_proxy is reused verbatim — the reuse the design calls "bottom-up".
@cell("logit_lens", family="llama", backend="hf")
def logit_lens_llama_hf(be, model, prompts, *, layers="all", unembed="module", residual="plain"):
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            model.model.layers, model.model.norm, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)


@cell("logit_lens", family="llama", backend="vllm_async")
def logit_lens_llama_vllm(be, model, prompts, *, layers="all", unembed="weight", residual="fused"):
    # residual="fused" is the backend-aware default: vLLM Llama uses fused-residual RMSNorm, so the
    # residual stream is hidden+residual. residual="plain" (naively ported from GPT-2) is SILENTLY_WRONG.
    def build():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _lens_proxy(
            model.model.layers, model.model.norm, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)


# --- nemotron family (NVIDIA Nemotron-H / Nemotron 3): a HYBRID stack where each block is a single
# op (Mamba-2 SSM | sparse attention | MLP | MoE), under `model.backbone.layers` / `.norm_f`, with an
# untied `lm_head` (§12.7). The residual stream is standard-additive across ALL block types
# (`hidden = residual + mixer_out`), and logit-lens is a residual-stream READ — so it ports unchanged:
# the lens does not care whether layer i is Mamba or attention. `_lens_proxy` is reused verbatim. The
# open frontier is whether nnsight-on-vLLM can trace NemotronH at all (its Mamba state lives in custom
# vLLM kernels), not the lens math.
@cell("logit_lens", family="nemotron", backend="hf")
def logit_lens_nemotron_hf(be, model, prompts, *, layers="all", unembed="module", residual="plain"):
    def build():
        return _lens_proxy(
            model.backbone.layers, model.backbone.norm_f, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)


@cell("logit_lens", family="nemotron", backend="vllm_async")
def logit_lens_nemotron_vllm(be, model, prompts, *, layers="all", unembed="weight", residual="plain"):
    # residual default "plain": HF NemotronH blocks already carry the full additive residual. Whether
    # vLLM's NemotronH impl fuses `(hidden, residual)` the way its Llama path does is UNMEASURED — the
    # spec runs both residual=plain and residual=fused tasks so a GPU sweep decides which is correct on
    # the vLLM side (§12.7).
    def build():
        return _lens_proxy(
            model.backbone.layers, model.backbone.norm_f, model.lm_head,
            layers=layers, unembed=unembed, last_fn=be.last, residual=residual,
        )
    return be.run(model, prompts, build)

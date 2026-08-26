"""Jacobian collection — fitting the jacobian lens's transport maps (the workload the existing
jacobian_lens spec only APPLIES).

Upstream estimator (anthropics/jacobian-lens, jlens/fitting.py): J_l is the average input-output
Jacobian dh_target/dh_l. Per prompt: one forward plus ceil(d_model / dim_batch) backward passes.
Each backward batches dim_batch copies of the prompt; copy b carries a one-hot cotangent on output
dimension (start + b) at EVERY valid target position at once (positions skip_first .. len-2; early
positions are attention sinks, the final position has no next-token target), so the gradient at
source position p is the sum over later targets, and J_l's row is the mean of that gradient over
the valid source positions. Rows accumulate across dim batches, J averages over prompts.

Structurally this is the `grad` frontier's third realization: not one backward per pair
(attribution) nor a training loop (DAS), but a many-VJP sweep with gradients read at EVERY source
layer per backward. HF runs it; on vLLM `requires_grad_` on the inference-tensor activations
raises inside `be.vjp_batch` -> ERROR.

The cell consumes the whole prompt list in one call (aggregate=False; J averages over prompts) and
returns the stacked maps [n_sources, d, d]. `grad=False` is the forward-only baseline (the overhead
denominator): the same batched forward, no cotangent, read the target residual's last row.
"""
from __future__ import annotations

import torch

from ..profiles import _resid
from .registry import cell
from .das import _d_model


def valid_slice(seq_len: int, skip_first: int) -> slice:
    """Positions included in the Jacobian average: skip the attention-sink prefix, drop the final
    position (no next-token target). Mirrors upstream's valid_position_mask."""
    if seq_len <= skip_first + 1:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return slice(skip_first, seq_len - 1)


def _collect_cell(be, model, m, prompts, *, dim_batch, skip_first, grad, residual):
    blocks = m.blocks(model)
    target = len(blocks) - 1
    sources = list(range(target))                 # J_l for every source layer below the target
    d = _d_model(model)

    def read_target():  # named so nnsight can source-serialize it to the vLLM worker
        with torch.no_grad():
            h = _resid(blocks[target].output, residual)
            return h.reshape(-1, h.shape[-1])[-1, :].clone()

    if not grad:        # baseline: the same forward, no VJP — the overhead denominator
        return be.run(model, [prompts[0]], read_target)

    def acts_of(mdl):   # named (not a lambda) so nnsight can source-serialize to the vLLM worker
        return [_resid(blocks[l].output, residual) for l in sources]

    def target_of(mdl):
        return _resid(blocks[target].output, residual)

    J = torch.zeros(len(sources), d, d)
    for prompt in prompts:
        for start in range(0, d, dim_batch):
            B = min(dim_batch, d - start)

            def make_cotangent(t):  # runs INSIDE the trace; t is the target residual —
                # [B, seq, d] on HF's replicated batch, [seq, d] on vLLM's single-prompt trace,
                # so the probe-row count comes from the tensor (per-dim when there is no batch)
                nb = t.shape[0] if t.dim() == 3 else 1
                flat = t.reshape(nb, -1, t.shape[-1])
                cot = torch.zeros_like(flat)
                pos = valid_slice(flat.shape[1], skip_first)
                for b in range(nb):                      # copy b probes output dim start+b
                    cot[b, pos, start + b] = 1.0
                return cot.reshape(t.shape)

            grads = be.vjp_batch(model, [prompt] * B, acts_of, target_of,
                                 make_cotangent, len(sources))
            for i, g in enumerate(grads):                # g: source-layer gradient, layout as above
                nb = g.shape[0] if g.dim() == 3 else 1
                gb = g.reshape(nb, -1, g.shape[-1])
                pos = valid_slice(gb.shape[1], skip_first)
                J[i, start:start + nb, :] += gb[:, pos, :].mean(dim=1)  # mean over source positions
    return J / len(prompts)


@cell("jacobian_collect", family="*", backend="hf")
def jacobian_collect_hf(be, model, m, prompts, *, dim_batch=96, skip_first=16,
                        grad=True, residual="plain"):
    return _collect_cell(be, model, m, prompts, dim_batch=dim_batch,
                         skip_first=skip_first, grad=grad, residual=residual)


@cell("jacobian_collect", family="*", backend="vllm_async")
def jacobian_collect_vllm(be, model, m, prompts, *, dim_batch=96, skip_first=16,
                          grad=True, residual="plain"):
    # Same explicit code as HF; the backward the sweep needs is unavailable on vLLM, so
    # be.vjp_batch surfaces the inference-tensor requires_grad error -> ERROR.
    return _collect_cell(be, model, m, prompts, dim_batch=dim_batch,
                         skip_first=skip_first, grad=grad, residual=residual)

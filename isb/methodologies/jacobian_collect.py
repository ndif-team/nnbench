"""Jacobian collection: fitting the Jacobian lens's transport maps (anthropics/jacobian-lens,
jlens/fitting.py). J_l is the average input-output Jacobian dh_target/dh_l: per prompt, one forward
plus ceil(d_model / dim_batch) backward passes, each over dim_batch copies of the prompt carrying a
one-hot cotangent on one output dimension at every valid target position (positions skip_first ..
len-2). The cell consumes the whole prompt list as one job and returns `[n_sources, d, d]`; HF runs
the sweep, vLLM's inference-mode activations refuse the backward. `grad=False` is the baseline: the
same forward over the units, reading the target residual's last row, `[B, d]`."""
from __future__ import annotations

import functools

import torch

from ..profiles import _resid
from .das import _d_model
from .observations import record_resolved
from .registry import cell


def valid_slice(seq_len: int, skip_first: int) -> slice:
    """Positions in the Jacobian average: after the attention-sink prefix, before the final position."""
    if seq_len <= skip_first + 1:
        raise ValueError(f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens")
    return slice(skip_first, seq_len - 1)


def _one_hot_cotangent(t, rows, *, start, skip_first):
    """Copy b of the target residual probes output dimension `start + rows[b]` at every valid
    position. `t` is `[B, seq, d]` (HF's replicated batch) or `[seq, d]` (one vLLM invoke)."""
    flat = t.reshape(len(rows), -1, t.shape[-1])
    cot = torch.zeros_like(flat)
    pos = valid_slice(flat.shape[1], skip_first)
    for b, row in enumerate(rows):
        cot[b, pos, start + row] = 1.0
    return cot.reshape(t.shape)


def _collect_cell(be, model, m, units, *, dim_batch, skip_first, grad, residual):
    blocks = m.blocks(model)
    target = len(blocks) - 1
    sources = list(range(target))
    d = _d_model(model)

    def read_target(rows):
        with torch.no_grad():
            return be.last(_resid(blocks[target].output, residual)).clone()

    if not grad:
        return be.run(model, units, read_target)

    def acts_of(mdl):
        return [_resid(blocks[l].output, residual) for l in sources]

    def target_of(mdl):
        return _resid(blocks[target].output, residual)

    J = torch.zeros(len(sources), d, d)

    def collect_rows(prompt, start, B):
        make_cotangent = functools.partial(_one_hot_cotangent, start=start, skip_first=skip_first)
        grads = be.vjp_batch(model, [prompt] * B, acts_of, target_of, make_cotangent, len(sources))
        for i, g in enumerate(grads):                                  # each [B, seq, d]
            pos = valid_slice(g.shape[1], skip_first)
            J[i, start:start + B, :] += g[:, pos, :].mean(dim=1)       # mean over source positions

    def collect():
        for prompt in units:
            for start in range(0, d, dim_batch):
                collect_rows(prompt, start, min(dim_batch, d - start))
        return J / len(units)

    return be.session(model, collect)


@cell("jacobian_collect", family="*", backend="hf")
def jacobian_collect_hf(be, model, m, units, *, dim_batch=96, skip_first=16, grad=True,
                        residual="plain"):
    return _collect_cell(be, model, m, units, dim_batch=dim_batch, skip_first=skip_first, grad=grad,
                         residual=residual)


@cell("jacobian_collect", family="*", backend="vllm_async", realizations=[{"residual": "plain"}])
def jacobian_collect_vllm(be, model, m, units, *, dim_batch=96, skip_first=16, grad=True,
                          residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    return _collect_cell(be, model, m, units, dim_batch=dim_batch, skip_first=skip_first, grad=grad,
                         residual=residual)

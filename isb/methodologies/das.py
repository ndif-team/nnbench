"""DAS, distributed alignment search (Geiger et al., 2023): an interchange intervention in a
rotated subspace at (layer, last position). `train=0` applies a seeded orthogonal rotation (the
same matrix on every backend); `train=N` fits the rotation with N AdamW steps over `batch` pairs
each (`be.train_patch`, whose backward crosses the frozen model: HF runs it, vLLM's inference-mode
activations refuse it). The cell consumes the whole labeled-unit list as one job inside one
`be.session`: it trains on all but the last `heldout` units and returns their patched final logits
`[heldout, vocab]`, evaluated in one patch call over the held-out units. In train mode the fitted
rotation's held-out interchange accuracy must beat the seeded rotation's, else the cell raises."""
from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .observations import record_resolved
from .readout import final_logits
from .registry import cell


def _d_model(model):
    for name in ("n_embd", "hidden_size"):
        if hasattr(model.config, name):
            return getattr(model.config, name)
    raise ValueError(f"cannot resolve d_model from config {type(model.config).__name__}")


def _seeded_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g))[0].contiguous()


def _rows(h):
    """[B, T, d] -> itself; vLLM's flat [T, d] -> [1, T, d]."""
    return h if h.dim() == 3 else h[None]


def _capture_last(blocks, layer, residual):
    """The source run's residual at (layer, last position), one row per prompt `[B, d]`."""
    with torch.no_grad():
        h = _resid(blocks[layer].output, residual)
        return _rows(h)[:, -1, :].clone()


def _interchange_read(blocks, model, head, *, layer, src, R, k, residual, grad, site):
    """Rotated interchange at (layer, last position) of every row, then the patched run's final
    logits `[B, vocab]`. `grad=True` keeps the graph alive for a loss on the logits."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        stream = _resid(out, residual)
        rows = _rows(stream)
        Rd = R.to(device=rows.device, dtype=rows.dtype)
        rot = rows[:, -1, :] @ Rd
        rot_src = src.to(device=rows.device, dtype=rows.dtype).reshape(-1, Rd.shape[0]) @ Rd
        mixed = torch.cat([rot_src[:, :k], rot[:, k:]], dim=-1)
        new = torch.cat([rows[:, :-1, :], (mixed @ Rd.T)[:, None, :]], dim=1).reshape(stream.shape)
        if residual == "fused":
            if not (is_tuple and len(out) >= 2):
                raise ValueError("fused DAS needs a (hidden, residual) layer output")
            blocks[layer].output = (new - out[1], *out[1:])
        else:
            blocks[layer].output = (new, *out[1:]) if is_tuple else new
        return final_logits(model, head, site)


def _das_cell(be, model, m, units, *, layer, k, train, batch, heldout, lr, seed, residual, site):
    if len(units) <= heldout:
        raise ValueError(f"need more than heldout={heldout} units, got {len(units)}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    train_units, eval_units = units[:-heldout], units[-heldout:]
    blocks, head = m.blocks(model), m.head(model)
    R = torch.nn.Parameter(_seeded_orthogonal(_d_model(model), seed))

    def answer_id(unit):
        """First token of the corrupted prompt's answer, with no special prefix (Llama-3's BOS)."""
        return model.tokenizer.encode(unit[2][1], add_special_tokens=False)[0]

    def capture():
        return _capture_last(blocks, layer, residual)

    def apply_fn(src):
        """The interchange read with the current rotation, no graph."""
        return _interchange_read(blocks, model, head, layer=layer, src=src, R=R.detach(), k=k,
                                 residual=residual, grad=False, site=site)

    def evaluate():
        """The held-out units' patched final logits `[heldout, vocab]`, corrupted runs as sources."""
        return be.patch(model, [u[1] for u in eval_units], [u[0] for u in eval_units],
                        capture=capture, patch=apply_fn)

    def iia():
        """Interchange accuracy on the held-out units (patched top-1 == counterfactual answer), and
        the logits it was measured on."""
        logits = evaluate()
        answers = torch.tensor([answer_id(u) for u in eval_units])
        return (logits.argmax(-1) == answers).float().mean().item(), logits

    def step(src, rows, *, targets, count):
        # the loss summed over the pairs this trace holds, divided by the pairs passed: the step's
        # mean whether one trace holds them all or one each
        logits = _interchange_read(blocks, model, head, layer=layer, src=src, R=R, k=k,
                                   residual=residual, grad=True, site=site)
        return F.cross_entropy(logits, targets[rows].to(logits.device), reduction="sum") / count

    def fit():
        if train:
            iia_seeded, _ = iia()
            opt = torch.optim.AdamW([R], lr=lr)
            for i in range(train):
                step_units = [train_units[(i * batch + j) % len(train_units)] for j in range(batch)]
                targets = torch.tensor([answer_id(u) for u in step_units])
                be.train_patch(model, [u[1] for u in step_units], [u[0] for u in step_units],
                               capture=capture,
                               step=functools.partial(step, targets=targets, count=len(step_units)))
                opt.step()
                opt.zero_grad()
            iia_trained, logits = iia()
            if iia_trained <= iia_seeded:
                raise RuntimeError(
                    f"DAS guard: trained rotation's held-out interchange accuracy "
                    f"({iia_trained:.2f}) does not beat the seeded rotation's ({iia_seeded:.2f})")
            return logits
        return evaluate()

    return be.session(model, fit)


@cell("das", family="*", backend="hf")
def das_hf(be, model, m, units, *, layer=6, k=64, train=0, batch=16, heldout=8,
           lr=1e-3, seed=0, residual="plain"):
    return _das_cell(be, model, m, units, layer=layer, k=k, train=train, batch=batch,
                     heldout=heldout, lr=lr, seed=seed, residual=residual, site="head")


@cell("das", family="*", backend="vllm_async", realizations=[{"residual": "plain"}])
def das_vllm(be, model, m, units, *, layer=6, k=64, train=0, batch=16, heldout=8,
             lr=1e-3, seed=0, residual=None):
    residual = residual if residual is not None else m.default_residual(be.name)
    record_resolved(residual=residual)
    return _das_cell(be, model, m, units, layer=layer, k=k, train=train, batch=batch,
                     heldout=heldout, lr=lr, seed=seed, residual=residual, site="engine")

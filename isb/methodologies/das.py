"""DAS — distributed alignment search (Geiger et al., 2023) — fixed per-cell methodology.

Interchange intervention in a ROTATED subspace: rotate the base run's residual at (layer, last
position) with an orthogonal matrix R, replace the first k rotated coordinates with the source
(counterfactual) run's, rotate back, let the forward continue, read the final logits. The
methodology has two realizations split by the `train` param:

- `train=0` (apply): R is a fixed seeded-orthogonal matrix (QR of a seeded Gaussian — the same
  matrix on both backends). COMPUTE (two matmuls) ∘ replacement-WRITE ∘ READ: the DAS program
  shape without the gradient loop, so it ports wherever activation patching does. A trained-R
  checkpoint would drop in the same way.
- `train=N` (the gradient loop): N AdamW steps on R, each over a minibatch of `batch` training
  pairs (16, the train split; pyvene's IOI sweep steps on 20, causalab on 32). Each step is one
  `be.train_patch` — capture the counterfactual activations (one row per pair), run the base batch
  with the rotated interchange at every row's last position, cross-entropy toward each pair's
  counterfactual answer token (the unit's stored label), backward. HF left-pads a causal batch
  (`TransformersModel._configure_tokenizer`), so `[:, -1]` is every row's real last token. The loss is measured
  at the model OUTPUT, so the backward necessarily crosses the frozen model's downstream layers to
  reach R — the property that makes DAS a `grad`-frontier workload: HF supports it; vLLM runs
  inference-mode tensors and the step raises -> ERROR (same class as attribution patching, now in
  the training-loop realization).

The cell consumes the WHOLE labeled-unit list in one call (aggregate=False): DAS trains one R
over a train split, then evaluates on held-out units. The fit runs inside one `be.session`, the
form nnsight documents for a multi-trace job (docs/request-patterns.md). Output = the held-out units' patched final
logits stacked [n_heldout, vocab] (comparable across backends for `train=0`, since R is seeded).
Non-vacuity guard (train mode): the trained R's interchange accuracy on the held-out units — the
fraction whose patched top-1 IS the counterfactual answer — must beat the seeded R's, else the
cell raises (a vacuous "training ran" would otherwise read as SUPPORTED).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..profiles import _resid
from .registry import cell


def _d_model(model):
    for name in ("n_embd", "hidden_size"):          # gpt2 / llama-shaped configs
        if hasattr(model.config, name):
            return getattr(model.config, name)
    raise ValueError(f"cannot resolve d_model from config {type(model.config).__name__}")


def _seeded_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g))[0].contiguous()


def _rows(h):
    """[B, T, d] -> itself; vLLM's flat single-prompt [T, d] -> [1, T, d]."""
    return h if h.dim() == 3 else h[None]


def _capture_last(blocks, layer, residual):
    """Trace-1 build: the source run's residual at (layer, last position), one row per prompt
    [B, d]. Runs in a trace."""
    with torch.no_grad():
        h = _resid(blocks[layer].output, residual)
        return _rows(h)[:, -1, :].clone()


def _interchange_read(blocks, norm, head, *, layer, src, R, k, residual, grad):
    """Rotated interchange at (layer, last position) of every row, then the patched run's final
    logits [B, vocab]. `src` is [B, d], one source row per prompt. Runs INSIDE a trace.
    `grad=False` wraps everything in no_grad (the apply/eval path); `grad=True` leaves the graph
    alive so a loss on the returned logits reaches R."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = blocks[layer].output
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        rows = _rows(h)                                         # [B, T, d]
        Rd = R.to(device=rows.device, dtype=rows.dtype)
        rot = rows[:, -1, :] @ Rd                               # rotate every row's last position
        rot_src = src.to(device=rows.device, dtype=rows.dtype).reshape(-1, Rd.shape[0]) @ Rd
        mixed = torch.cat([rot_src[:, :k], rot[:, k:]], dim=-1)  # swap the first k coordinates
        new = torch.cat([rows[:, :-1, :], (mixed @ Rd.T)[:, None, :]], dim=1).reshape(h.shape)
        blocks[layer].output = (new, *out[1:]) if is_tuple else new  # whole-tuple replace

        normed = norm(_resid(blocks[-1].output, residual))      # patched run's final readout
        logits = F.linear(normed, head.weight)                  # portable unembed
        return _rows(logits)[:, -1, :]


def _das_cell(be, model, m, prompts, *, layer, k, train, batch, heldout, lr, seed, residual):
    units = list(prompts)                    # labeled units: (clean, corrupted, (correct, incorrect))
    if len(units) <= heldout:
        raise ValueError(f"need more than heldout={heldout} units, got {len(units)}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    train_units, eval_units = units[:-heldout], units[-heldout:]
    blocks, ln_f, head = m.blocks(model), m.norm(model), m.head(model)
    R = torch.nn.Parameter(_seeded_orthogonal(_d_model(model), seed))

    def answer_id(unit):
        return model.tokenizer.encode(unit[2][1])[0]            # the corrupted prompt's answer

    def fit():
        if train:
            iia_seeded = iia(R)
            opt = torch.optim.AdamW([R], lr=lr)
            for i in range(train):
                # minibatch i: `batch` consecutive training pairs, wrapping around the split
                step_units = [train_units[(i * batch + j) % len(train_units)] for j in range(batch)]
                clean = [u[0] for u in step_units]
                corrupted = [u[1] for u in step_units]
                targets = torch.tensor([answer_id(u) for u in step_units])

                def step(src, rows):  # loss at the OUTPUT of the intervened forward -> backward reaches R
                    # `rows`: the step's pairs in this trace (all of them in HF's padded batch, one
                    # per trace on the async engine); summed over them (train_patch's contract)
                    # and divided by the pairs passed, so the step's loss is their mean
                    logits = _interchange_read(blocks, ln_f, head, layer=layer, src=src,
                                               R=R, k=k, residual=residual, grad=True)
                    return F.cross_entropy(logits, targets[rows].to(logits.device),
                                           reduction="sum") / len(step_units)
                be.train_patch(model, corrupted, clean, capture=capture, step=step)
                opt.step()
                opt.zero_grad()
            iia_trained = iia(R)
            if iia_trained <= iia_seeded:        # the non-vacuity guard (approved held-out check)
                raise RuntimeError(
                    f"DAS guard: trained rotation's held-out interchange accuracy "
                    f"({iia_trained:.2f}) does not beat the seeded rotation's ({iia_seeded:.2f})")
        return torch.stack([eval_dist(u, R.detach())[0] for u in eval_units])

    def capture():  # named (not a lambda) so nnsight can source-serialize it to the vLLM worker
        return _capture_last(blocks, layer, residual)

    def eval_dist(unit, rot):
        clean, corrupted, _answers = unit

        def apply_fn(src):  # rotated interchange with a FIXED rotation, no grad
            return _interchange_read(blocks, ln_f, head, layer=layer, src=src,
                                     R=rot, k=k, residual=residual, grad=False)
        return be.patch(model, corrupted, clean, capture=capture, patch=apply_fn)

    def iia(rot):
        """Interchange accuracy on the held-out units: patched top-1 == counterfactual answer."""
        hits = 0
        for unit in eval_units:
            hits += int(eval_dist(unit, rot.detach())[0].argmax().item() == answer_id(unit))
        return hits / len(eval_units)

    return be.session(model, fit)            # the whole fit is one multi-trace job


@cell("das", family="*", backend="hf")
def das_hf(be, model, m, prompts, *, layer=6, k=64, train=0, batch=16, heldout=8,
           lr=1e-3, seed=0, residual="plain"):
    return _das_cell(be, model, m, prompts, layer=layer, k=k, train=train, batch=batch,
                     heldout=heldout, lr=lr, seed=seed, residual=residual)


@cell("das", family="*", backend="vllm_async")
def das_vllm(be, model, m, prompts, *, layer=6, k=64, train=0, batch=16, heldout=8,
             lr=1e-3, seed=0, residual="plain"):
    # Same explicit code as HF. `train=0` is COMPUTE ∘ replacement-WRITE and runs; `train>0`
    # needs the backward, which the backend's train_patch surfaces as the inference-tensor
    # requires_grad error — the `grad` frontier in its training-loop realization.
    return _das_cell(be, model, m, prompts, layer=layer, k=k, train=train, batch=batch,
                     heldout=heldout, lr=lr, seed=seed, residual=residual)

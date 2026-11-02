"""nnsight 0.8 cells in the fastest documented realization (backend `nnsight_opt`).

Same semantics as the general nnsight cells (isb/methodologies/): the fused-residual stream is
`out[0] + out[1]` on vLLM's Qwen2 layers, steering adds `alpha * scale * d` with `d` the target's
unit unembedding row, ablation zeroes a submodule's output, readout is the next-token logits.
Realization differs (docs/models/vllm-editing.md, vllm.md):

- per-task-fixed blocks are engine edits installed once (backend_opt.py, `NnsightOptBackend`),
  selected by name on each plain request over all units of a call, and read logits from the
  engine site `model.logits`;
- patching keeps two traces (the clean value changes per call), one invoke per unit, whose blocks
  reference only the layer handles they touch, never the root model;
- writes are in place; reads that outlive their step are cloned (they alias engine memory).

Tensor parallelism is transparent to the blocks (docs/models/vllm-parallelism.md) except for
parameters, which are this rank's slice; so the lens projects through `model.logits_processor`
(vllm-parallelism.md's form) and the steering direction is the checkpoint's unembedding row
(`readout.checkpoint_direction`), built before the edit and moved onto the stream inside it.

A `with` block must sit in the cell's own frame: nnsight compiles the block from that frame.
Registered for family "llama" (vLLM Qwen2 paths `model.layers`, `model.norm`, `lm_head`). Outputs
carry the unit axis at dim -2.
"""
import functools

import torch

from isb.methodologies.readout import checkpoint_direction, checkpoint_of
from isb.methodologies.registry import cell
from isb.methodologies.steering import (CALIBRATED, _mean_norms, _resolve_token,
                                      calibrated_strengths)

EDIT = "isb-cell"


def _ensure(be, key):
    """True when the edit for `key` is already installed; otherwise clear whatever is."""
    if be.edit_installed(key):
        return True
    be.clear_edit()
    return False


def _steps(rows, new_tokens):
    """One request's per-step rows -> `[new_tokens, vocab]`."""
    if len(rows) != new_tokens:
        raise RuntimeError(f"collected {len(rows)} per-step rows, expected {new_tokens}")
    return torch.cat([r.float().cpu() for r in rows], dim=0)


@cell("logit_lens", family="llama", backend="nnsight_opt")
def logit_lens(be, model, units, *, layers="all"):
    blocks = model.model.layers
    n = len(blocks)
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    key = ("logit_lens", tuple(idx))
    if not _ensure(be, key):
        norm, head = model.model.norm, model.lm_head
        with model.edit(name=EDIT) as (tracer, edit):
            rows = []
            for i in idx:
                out = blocks[i].output
                rows.append(out[0][-1:] + out[1][-1:])          # last-token fused stream
            lens = model.logits_processor(head, norm(torch.cat(rows))).save()   # one batched readout
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    return torch.stack([s["lens"].float().cpu() for s in saves], dim=1)     # [n_layers, B, vocab]


@cell("residual_readout", family="llama", backend="nnsight_opt")
def residual_readout(be, model, units, *, layers):
    """The last-token fused stream `out[0] + out[1]` after each listed layer, concatenated in the
    worker and copied to host memory there, in the engine's dtype: one save per request."""
    blocks = model.model.layers
    n = len(blocks)
    idx = [i % n for i in layers]
    key = ("residual_readout", tuple(idx))
    if not _ensure(be, key):
        with model.edit(name=EDIT) as (tracer, edit):
            rows = []
            for i in idx:
                out = blocks[i].output
                rows.append(out[0][-1:] + out[1][-1:])          # last-token fused stream
            stream = torch.cat(rows).cpu().save()               # [n_layers, d]
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    return torch.stack([s["stream"] for s in saves], dim=1)     # [n_layers, B, d]


@cell("activation_harvest", family="llama", backend="nnsight_opt")
def activation_harvest(be, model, units, *, layer=None, export=True, residual=None, layers=None):
    from isb.methodologies.activation_harvest import _layers
    from isb.methodologies.observations import record_resolved

    record_resolved(residual="fused")
    read = _layers(layer, layers, export)
    blocks = model.model.layers
    n = len(blocks)
    idx = None if read is None else tuple(i % n for i in read)
    key = ("activation_harvest", layer, export, idx)
    if not _ensure(be, key):
        with model.edit(name=EDIT) as (tracer, edit):
            if not export:
                acts = model.logits[-1:, :].argmax(-1, keepdim=True).cpu().save()
            elif idx is None:
                out = blocks[layer % n].output
                acts = (out[0] + out[1]).cpu().save()
            else:
                acts = torch.stack([blocks[i].output[0] + blocks[i].output[1]
                                    for i in idx]).cpu().save()
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    if not export or idx is None:
        return torch.cat([s["acts"] for s in saves], dim=0)
    return torch.cat([s["acts"] for s in saves], dim=1)


@cell("gen_capture", family="llama", backend="nnsight_opt")
def gen_capture(be, model, units, *, layer, new_tokens, export=True, bound="bounded",
                residual=None):
    import nnsight
    from isb.methodologies.gen_steering import _check_bound
    from isb.methodologies.observations import record_resolved

    _check_bound(bound)
    record_resolved(residual="fused")
    key = ("gen_capture", layer, export)
    if not _ensure(be, key):
        block = model.model.layers[layer]
        with model.edit(name=EDIT) as (tracer, edit):
            rows = nnsight.save([])
            for _ in tracer.all():
                if not export:
                    rows.append(model.logits[-1:, :].argmax(-1, keepdim=True).cpu())
                else:
                    out = block.output
                    rows.append((out[0][-1:] + out[1][-1:]).cpu())
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units, max_tokens=new_tokens)
    return torch.stack([torch.cat(s["rows"], dim=0) for s in saves], dim=1)


@cell("steering", family="llama", backend="nnsight_opt")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm",
             layers=None):
    """`layers`, when given, applies the edit at each listed layer in forward order and replaces
    `layer`, each layer getting `alpha / len(layers)`; the direction moves onto the engine's device
    once per request. `scale_by="calibrated_mean_norm"` adds fixed vectors whose norms one plain
    read measures before the edit is installed (isb/methodologies/steering.py,
    `calibrated_strengths`), so the installed edit reads no live norm."""
    idx = (layer,) if layers is None else tuple(layers)
    key = ("steering", idx, target, alpha, scale_by)
    if not _ensure(be, key):
        unit = checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
        blocks = model.model.layers
        targets = [blocks[i] for i in idx]
        share = alpha / len(idx)
        strengths = None
        if scale_by == CALIBRATED and alpha != 0:       # no edit is installed during the read
            measure = functools.partial(_mean_norms, be, model, blocks, residual="fused",
                                        masked=False)
            strengths = calibrated_strengths(be, list(idx), alpha, units, measure)
        with model.edit(name=EDIT) as (tracer, edit):
            if alpha != 0:
                direction = None
                for k, block in enumerate(targets):
                    out = block.output
                    if direction is None:
                        direction = unit.to(device=out[0].device, dtype=out[0].dtype)
                    if strengths is not None:
                        out[0][:] += strengths[k] * direction             # in place, fixed
                    else:
                        stream = out[0] + out[1]
                        if scale_by == "mean_norm":
                            scale = stream.norm(dim=-1).mean()
                        else:
                            scale = stream.norm(dim=-1, keepdim=True)
                        out[0][:] += (share * scale) * direction          # in place
            logits = model.logits.save()
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    return torch.cat([s["logits"][-1:].float().cpu() for s in saves])        # [B, vocab]


@cell("gen_steering", family="llama", backend="nnsight_opt")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    key = ("gen_steering", layer, target, alpha, scale_by)
    if not _ensure(be, key):
        import nnsight

        unit = checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
        block = model.model.layers[layer]
        with model.edit(name=EDIT) as (tracer, edit):
            rows = nnsight.save([])
            for _ in tracer.all():                                # prefill and every decode step
                if alpha != 0:
                    out = block.output
                    direction = unit.to(device=out[0].device, dtype=out[0].dtype)
                    stream = out[0] + out[1]
                    if scale_by == "mean_norm":
                        scale = stream.norm(dim=-1).mean()
                    else:
                        scale = stream.norm(dim=-1, keepdim=True)
                    out[0][:] += (alpha * scale) * direction
                rows.append(model.logits[-1:].clone())
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units, max_tokens=new_tokens)
    return torch.stack([_steps(s["rows"], new_tokens) for s in saves], dim=1)   # [new_tokens, B, vocab]


@cell("activation_patching", family="llama", backend="nnsight_opt")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all"):
    be.clear_edit()                    # traces only; no installed block may ride these requests
    clean_prompts = [unit[0] for unit in units]
    corrupt_prompts = [unit[1] for unit in units]
    block = model.model.layers[layer]
    n = len(units)
    sampling = {"temperature": 0.0, "top_p": 1, "max_tokens": 1, "edits": []}
    clean = [None] * n
    if patch:
        with model.trace(**sampling) as tracer:
            clean = clean.save()
            for i, prompt in enumerate(clean_prompts):
                with tracer.invoke(prompt):
                    out = block.output
                    clean[i] = (out[0] + out[1]).clone()         # the clean stream, bf16
    with model.trace(**sampling) as tracer:
        rows = [None] * n
        rows = rows.save()
        for i, prompt in enumerate(corrupt_prompts):
            with tracer.invoke(prompt):
                if patch:
                    out = block.output
                    value = clean[i].to(out[0].device, out[0].dtype)
                    if positions == "last":
                        out[0][-1:] = value[-1:] - out[1][-1:]
                    else:
                        out[0][:] = value - out[1]              # stream := clean, in place
                rows[i] = model.logits[-1:]                     # the engine site, as below
    return torch.cat([r.float().cpu() for r in rows])            # [B, vocab]


@cell("ablation", family="llama", backend="nnsight_opt")
def ablation(be, model, units, *, layer=6, target="mlp"):
    key = ("ablation", layer, target)
    if not _ensure(be, key):
        block = model.model.layers[layer]
        sub = block.mlp if target in ("mlp", "none") else block.self_attn
        with model.edit(name=EDIT) as (tracer, edit):
            if target != "none":
                out = sub.output
                if isinstance(out, tuple):
                    out[0][:] = 0
                else:
                    out[:] = 0                                    # in place
            logits = model.logits.save()
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    return torch.cat([s["logits"][-1:].float().cpu() for s in saves])        # [B, vocab]


@cell("jacobian_lens", family="*", backend="nnsight_opt")
def jacobian_lens(be, model, m, units, *, layers="all", position="last", residual=None,
                  transport=None):
    """The J-lens readout `logits_processor(head, norm(J_l @ h_l))` as an installed edit.

    The general cell (isb/methodologies/jacobian_lens.py) is a trace, and a trace's block is
    serialized onto every request with what it references (vllm-editing.md, "When to edit instead
    of trace"); for the fitted Qwen3.5-4B lens that is the 31 transport maps on each request. Here
    the maps go to the engine once, with the edit: stacked `[n_layers, d, d]` (one `[d, d]` for a
    seeded map) in the engine's dtype and placed on the GPU before the edit is installed, so the
    worker unpickles them onto its device once (`GPUModelRunner.Requests.register` deserializes an
    edit once; each request's copy shares its locals) and a request moves no transport bytes.

    Sites, as in the general cell: a vLLM decoder layer's output is `(hidden, residual)` whose sum
    is the stream after the layer ("fused", vllm.md "What your block sees on vLLM"); the row at the
    readout position goes through one batched matmul, the final norm and the logits processor once
    per request, and the `[n_layers, vocab]` lens comes back on the host (`.cpu()` in the block,
    vllm.md). An edit carries no per-request value, so every unit must read the same position
    (`last` is -1 in every prompt). The profile view is the vLLM one: this backend is nnsight's sync
    vLLM engine, so the module paths carry the family's vLLM prefix."""
    from collections.abc import Mapping

    from isb.methodologies.jacobian_lens import _load_fitted, _orthogonal
    from isb.methodologies.observations import record_resolved
    from isb.profiles import _resid
    from isb.tasks.lens_eval import locate

    view = m.for_backend("vllm_sync")
    residual = residual if residual is not None else view.default_residual("vllm_sync")
    record_resolved(residual=residual)
    positions = {locate(model.tokenizer, unit, position) for unit in units}
    if len(positions) != 1:
        raise ValueError(f"an installed edit reads one position for every request; position "
                         f"{position!r} resolves to {sorted(positions)} over these units")
    pos = positions.pop()
    blocks = view.blocks(model)
    n = len(blocks)
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    spec = transport if transport is None or isinstance(transport, (int, str)) else id(transport)
    key = ("jacobian_lens", tuple(idx), pos, residual, spec)
    if not _ensure(be, key):
        norm, head, processor = view.norm(model), view.head(model), view.logits_processor(model)
        dtype = head.weight.dtype                          # the engine's dtype (the meta tree's)
        maps = _load_fitted(transport) if isinstance(transport, str) else transport
        if maps is None:
            J = None
        elif isinstance(maps, Mapping):
            J = torch.stack([maps[i] for i in idx])        # [n_layers, d, d]
        elif isinstance(maps, torch.Tensor):
            J = maps
        else:
            J = _orthogonal(head.weight.shape[-1], maps)   # one seeded [d, d] for every layer
        if J is not None:
            J = J.to(device="cuda", dtype=dtype)
        with model.edit(name=EDIT) as (tracer, edit):
            rows = []
            for i in idx:
                out = blocks[i].output
                rows.append(out[0][pos] + out[1][pos] if residual == "fused"
                            else _resid(out, residual)[pos])
            h = torch.stack(rows)                                          # [n_layers, d]
            if J is not None:
                h = (torch.bmm(J, h.unsqueeze(-1)).squeeze(-1) if J.dim() == 3
                     else h @ J.T)                                         # J_l @ h_l per layer
            lens = processor(head, norm(h)).cpu().save()                   # [n_layers, vocab]
        be.remember_edit(key, edit)
    saves = be.run_edit(model, units)
    return torch.stack([s["lens"].float() for s in saves], dim=1)         # [n_layers, B, vocab]

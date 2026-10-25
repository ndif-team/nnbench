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
import torch

from isb.methodologies.readout import checkpoint_direction, checkpoint_of
from isb.methodologies.registry import cell
from isb.methodologies.steering import _resolve_token

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


@cell("steering", family="llama", backend="nnsight_opt")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    key = ("steering", layer, target, alpha, scale_by)
    if not _ensure(be, key):
        unit = checkpoint_direction(checkpoint_of(model), _resolve_token(model.tokenizer, target))
        block = model.model.layers[layer]
        with model.edit(name=EDIT) as (tracer, edit):
            if alpha != 0:
                out = block.output
                direction = unit.to(device=out[0].device, dtype=out[0].dtype)
                stream = out[0] + out[1]
                if scale_by == "mean_norm":
                    scale = stream.norm(dim=-1).mean()
                else:
                    scale = stream.norm(dim=-1, keepdim=True)
                out[0][:] += (alpha * scale) * direction          # in place
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

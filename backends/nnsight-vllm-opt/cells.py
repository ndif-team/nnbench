"""nnsight 0.8 cells in the fastest documented realization (backend `nnsight_opt`).

Same semantics as the general nnsight cells (isb/methodologies/): the fused-residual stream is
`out[0] + out[1]` on vLLM's Qwen2 layers, steering adds `alpha * scale * d` with `d` the target's
unit unembedding row, ablation zeroes a submodule's output, readout is the next-token logits.
Realization differs (docs/models/vllm-editing.md, vllm.md):

- per-task-fixed blocks are engine edits installed once (`run.py`, `NnsightOptBackend`), selected
  by name on each plain request, and read logits from the engine site `model.logits`;
- patching keeps two traces (the clean value changes per call) whose blocks reference only the
  layer handles they touch, never the root model;
- writes are in place; reads that outlive their step are cloned (they alias engine memory).

A `with` block must sit in the cell's own frame: nnsight compiles the block from that frame.
Registered for family "llama" (vLLM Qwen2 paths `model.layers`, `model.norm`, `lm_head`).
"""
import torch
import torch.nn.functional as F

from isb.methodologies.registry import cell
from isb.methodologies.steering import _resolve_token

EDIT = "isb-cell"


def _ensure(be, key):
    """True when the edit for `key` is already installed; otherwise clear whatever is."""
    if be.edit_installed(key):
        return True
    be.clear_edit()
    return False


@cell("logit_lens", family="llama", backend="nnsight_opt")
def logit_lens(be, model, prompts, *, layers="all"):
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
            lens = F.linear(norm(torch.cat(rows)), head.weight).save()   # one batched readout
        be.remember_edit(key, edit)
    saves = be.run_edit(model, prompts[0])
    return saves["lens"].float().cpu().unsqueeze(1)               # [n_layers, 1, vocab]


@cell("steering", family="llama", backend="nnsight_opt")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    key = ("steering", layer, target, alpha, scale_by)
    if not _ensure(be, key):
        token_id = _resolve_token(model.tokenizer, target)
        block, head = model.model.layers[layer], model.lm_head
        with model.edit(name=EDIT) as (tracer, edit):
            if alpha != 0:
                out = block.output
                direction = F.normalize(head.weight[token_id].float(), dim=0).to(out[0].dtype)
                stream = out[0] + out[1]
                if scale_by == "mean_norm":
                    scale = stream.norm(dim=-1).mean()
                else:
                    scale = stream.norm(dim=-1, keepdim=True)
                out[0][:] += (alpha * scale) * direction          # in place
            logits = model.logits.save()
        be.remember_edit(key, edit)
    saves = be.run_edit(model, prompts[0])
    return saves["logits"][-1:].float().cpu()                     # [1, vocab]


@cell("gen_steering", family="llama", backend="nnsight_opt")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    key = ("gen_steering", layer, target, alpha, scale_by)
    if not _ensure(be, key):
        import nnsight

        token_id = _resolve_token(model.tokenizer, target)
        block, head = model.model.layers[layer], model.lm_head
        with model.edit(name=EDIT) as (tracer, edit):
            rows = nnsight.save([])
            for _ in tracer.all():                                # prefill and every decode step
                if alpha != 0:
                    out = block.output
                    direction = F.normalize(head.weight[token_id].float(), dim=0).to(out[0].dtype)
                    stream = out[0] + out[1]
                    if scale_by == "mean_norm":
                        scale = stream.norm(dim=-1).mean()
                    else:
                        scale = stream.norm(dim=-1, keepdim=True)
                    out[0][:] += (alpha * scale) * direction
                rows.append(model.logits[-1:].clone())
        be.remember_edit(key, edit)
    saves = be.run_edit(model, prompts[0], max_tokens=new_tokens)
    rows = saves["rows"]
    if len(rows) != new_tokens:
        raise RuntimeError(f"collected {len(rows)} per-step rows, expected {new_tokens}")
    return torch.cat([r.float().cpu() for r in rows], dim=0)      # [new_tokens, vocab]


@cell("activation_patching", family="llama", backend="nnsight_opt")
def activation_patching(be, model, prompts, *, layer=6, patch=True, positions="all"):
    be.clear_edit()                    # traces only; no installed block may ride these requests
    clean_prompt, corrupt_prompt = prompts
    block, final = model.model.layers[layer], model.model.layers[-1]
    norm, head = model.model.norm, model.lm_head
    sampling = {"temperature": 0.0, "top_p": 1, "max_tokens": 1, "edits": []}
    if patch:
        with model.trace(clean_prompt, **sampling):
            out = block.output
            clean = (out[0] + out[1]).clone().save()             # the clean stream, bf16
    else:
        clean = None
    with model.trace(corrupt_prompt, **sampling):
        if clean is not None:
            out = block.output
            if positions == "last":
                out[0][-1:] = clean[-1:].to(out[0].dtype) - out[1][-1:]
            else:
                out[0][:] = clean.to(out[0].dtype) - out[1]      # stream := clean, in place
        fin = final.output
        logits = F.linear(norm(fin[0][-1:] + fin[1][-1:]), head.weight).save()
    return logits.float().cpu()                                   # [1, vocab]


@cell("ablation", family="llama", backend="nnsight_opt")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
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
    saves = be.run_edit(model, prompts[0])
    return saves["logits"][-1:].float().cpu()

"""nnsight 0.4.6 cells: the same rows as isb/methodologies/ (same observable, same strengths, same
readout) spelled in 0.4's graph API. A module output's structure is fixed before the trace from the
backend's arity probe (`be.arity`): a tuple output is read at `[0]` and replaced by a tuple of the
same length, a bare tensor is read and replaced whole. Python control flow on values runs on
parameters, never on proxies. Writes follow the 0.4 README: assignment to `.output` replaces it,
`[:]` writes in place. Outputs carry the unit axis at dim -2.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isb.methodologies.registry import cell
from isb.methodologies.steering import _resolve_token

BACKEND = "hf_nnsight_0_4"


def _read(out, arity):
    return out[0] if arity else out


def _replace(envoy, out, arity, value):
    """Assign `value` as the stream of `envoy`'s output, keeping any other tuple members."""
    envoy.output = (value, *(out[i] for i in range(1, arity))) if arity else value


def _mask(model, hidden):
    """The batch's attention mask over `hidden`'s positions: the root module's `attention_mask`
    keyword (`model.inputs` is `(args, kwargs)`), its newest columns when a decode step serves one
    position. `narrow` takes the proxy offsets that a slice could not carry."""
    mask = model.inputs[1]["attention_mask"]
    width = hidden.shape[-2]
    return torch.narrow(mask, -1, mask.shape[-1] - width, width)


def _scale(stream, scale_by, mask):
    """Steering strength per unit: the mean norm over each unit's real tokens, `[B, 1, 1]`, or each
    token's own norm, `[B, S, 1]`."""
    if scale_by == "mean_norm":
        m = mask.to(stream.dtype)
        return ((stream.norm(dim=-1) * m).sum(-1) / m.sum(-1))[:, None, None]
    if scale_by == "token_norm":
        return stream.norm(dim=-1, keepdim=True)
    raise ValueError(f"unknown steering scale {scale_by!r} (expected 'mean_norm' or 'token_norm')")


def _final_logits(head):
    """The model's own next-token logits, last token (isb/methodologies/readout.py, "head")."""
    return head.output[:, -1, :]


def _device(envoy):
    """Where a module's weights live: under `device_map` the layers sit on different GPUs. 0.4
    cannot move a real tensor to a proxy's device (torch 2.6 rejects proxy arguments), so a vector
    built outside the trace goes to its target module's device before the trace."""
    return next(envoy._module.parameters()).device


def _direction(head, token_id, envoy):
    """Unit unembedding row of `token_id`, built before the trace, on `envoy`'s device."""
    row = head._module.weight[token_id].detach().float()
    return F.normalize(row, dim=0).to(device=_device(envoy), dtype=head._module.weight.dtype)


@cell("logit_lens", family="*", backend=BACKEND, realizations=[{"unembed": "weight"}])
def logit_lens(be, model, m, units, *, layers="all", unembed="module"):
    blocks, norm, head = m.blocks(model), m.norm(model), m.head(model)
    idx = list(range(len(blocks))) if layers == "all" else list(layers)
    arity = be.arity(model, [blocks[i] for i in idx])

    def build(rows):
        lens = []
        for i, a in zip(idx, arity):
            normed = norm(be.last(_read(blocks[i].output, a)))       # last token first
            lens.append(F.linear(normed, head.weight) if unembed == "weight" else head(normed))
        return torch.stack(lens, dim=0)                              # [n_layers, B, vocab]

    return be.run(model, units, build)


@cell("steering", family="*", backend=BACKEND, realizations=[{"mode": "inplace"}])
def steering(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0, mode="replace",
             scale_by="mean_norm"):
    blocks, head = m.blocks(model), m.head(model)
    n = len(blocks)
    if alpha != 0 and layer % n == n - 1:
        raise ValueError(f"steer layer {layer} is the read-out block; pick layer < {n - 1}")
    if mode not in ("inplace", "replace"):
        raise ValueError(f"unknown steering mode {mode!r}")
    direction = _direction(head, _resolve_token(model.tokenizer, target), blocks[layer])
    (a_layer,) = be.arity(model, [blocks[layer]])

    def build(rows):
        if alpha != 0:
            out = blocks[layer].output
            hidden = _read(out, a_layer)
            vec = (alpha * _scale(hidden, scale_by, _mask(model, hidden))) * direction
            if mode == "inplace":
                hidden[:] = hidden + vec
            else:
                _replace(blocks[layer], out, a_layer, hidden + vec)
        return _final_logits(head)

    return be.run(model, units, build)


@cell("ablation", family="*", backend=BACKEND)
def ablation(be, model, m, units, *, layer=6, target="mlp"):
    blocks, head = m.blocks(model), m.head(model)
    module = m.submodule(blocks[layer], "mlp" if target == "none" else target)
    (a_target,) = be.arity(model, [module])

    def build(rows):
        if target != "none":
            out = module.output
            _replace(module, out, a_target, torch.zeros_like(_read(out, a_target)))
        return _final_logits(head)

    return be.run(model, units, build)


@cell("activation_patching", family="*", backend=BACKEND)
def activation_patching(be, model, m, units, *, layer=6, patch=True, positions="all"):
    clean = [unit[0] for unit in units]
    corrupted = [unit[1] for unit in units]
    blocks, head = m.blocks(model), m.head(model)
    (a_layer,) = be.arity(model, [blocks[layer]])
    if not patch:
        def build(rows):
            return _final_logits(head)
        return be.run(model, corrupted, build)
    if positions not in ("all", "last"):
        raise ValueError(f"unknown patch positions {positions!r} (expected 'all' or 'last')")
    tokenize = model.tokenizer
    for clean_prompt, corrupted_prompt in zip(clean, corrupted):
        if len(tokenize(clean_prompt)["input_ids"]) != len(tokenize(corrupted_prompt)["input_ids"]):
            raise ValueError("clean/corrupted token lengths must match for position-aligned patching")

    # the snapshot is a real tensor: move it with real values (torch 2.6 rejects proxy arguments)
    dtype = next(model._model.parameters()).dtype
    device = _device(blocks[layer])                  # the patched layer's GPU under device_map

    def capture():
        return _read(blocks[layer].output, a_layer).clone()

    def patch_fn(clean_act):
        out = blocks[layer].output
        hidden = _read(out, a_layer)
        value = clean_act.to(device=device, dtype=dtype)
        if positions == "last":
            value = torch.cat([hidden[..., :-1, :], value[..., -1:, :]], dim=-2)
        _replace(blocks[layer], out, a_layer, value)
        return _final_logits(head)

    return be.patch(model, clean, corrupted, capture=capture, patch=patch_fn)


@cell("gen_steering", family="*", backend=BACKEND)
def gen_steering(be, model, m, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    blocks, head = m.blocks(model), m.head(model)
    direction = _direction(head, _resolve_token(model.tokenizer, target), blocks[layer])
    (a_layer,) = be.arity(model, [blocks[layer]])

    def step(index):
        if alpha != 0:
            out = blocks[layer].output
            hidden = _read(out, a_layer)
            vec = (alpha * _scale(hidden, scale_by, _mask(model, hidden))) * direction
            _replace(blocks[layer], out, a_layer, hidden + vec)
        return head.output[:, -1, :]

    return be.generate(model, units, step, new_tokens=new_tokens)

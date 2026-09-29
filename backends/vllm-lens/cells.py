"""vLLM-Lens cells for the comparison specs (design.md §12.14).

Each cell expresses the spec's semantics with vLLM-Lens 1.2.1's own mechanisms: `Hook` functions on
decoder layers (post-hooks see the summed residual stream, and a returned tensor replaces it) and
`output_residual_stream` capture. Readouts run on the worker inside a hook, as in vLLM-Lens's
logit-lens example: the model's final norm, then the unembedding weight from `ctx.get_parameter`.
Outputs match the nnsight cells' shapes: `[1, vocab]` per forward readout.

Registered for family "llama" (vLLM module paths `model.norm`, `lm_head`; vLLM-Lens itself finds
decoder layers only under `model.layers`-style trees).
"""
import sys

import cloudpickle
import torch
import torch.nn.functional as F

from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import _resolve_token

# The hook closures travel to the engine-core process by value; this module is not on its path.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

_GREEDY = {"temperature": 0.0, "top_p": 1.0}


def _readout(ctx, h):
    """Next-token logits of the last row: final norm, then the unembedding. Runs on the worker."""
    normed = ctx.model.model.norm(h[-1:])
    return F.linear(normed, ctx.get_parameter("lm_head.weight")).float().cpu()


def _hooked(model, prompt, fn, layers, *, max_tokens=1, extra=None):
    """One request with one per-request Hook; returns the hook's saved dict."""
    from vllm import SamplingParams
    from vllm_lens import Hook

    params = SamplingParams(max_tokens=max_tokens, **_GREEDY, extra_args={
        **(extra or {}), "apply_hooks": [Hook(fn=fn, layer_indices=sorted(set(layers)))]})
    (out,) = model.llm.generate([prompt], params, use_tqdm=False)
    return out.hook_results["0"]


def _rows(saved, expected):
    rows = saved.get("rows", [])
    if len(rows) != expected:     # a chunked prefill or an early EOS would change the count
        raise RuntimeError(f"collected {len(rows)} readout rows, expected {expected}")
    return rows


def _steer_fn(layer, token_id, alpha, last):
    """Add `alpha` times the residual's mean per-token norm along the target's unembedding row at
    `layer`; read the next-token logits at the last layer. Fires on every forward."""
    def steer(ctx, h):
        if ctx.layer_idx == layer and alpha != 0:
            weight = ctx.get_parameter("lm_head.weight")
            direction = F.normalize(weight[token_id].float(), dim=0).to(weight.dtype)
            return h + (alpha * h.norm(dim=-1).mean()) * direction
        if ctx.layer_idx == last:
            ctx.saved.setdefault("rows", []).append(_readout(ctx, h))
        return None
    return steer


def _direction(model, token_id):
    """The target token's unit unembedding row, fetched once from the worker through a hook."""
    if token_id not in model.directions:
        def fetch(ctx, h):
            ctx.saved["row"] = ctx.get_parameter("lm_head.weight")[token_id].float().cpu()
        row = _hooked(model, " ", fetch, [0])["row"]
        model.directions[token_id] = F.normalize(row, dim=0)
    return model.directions[token_id]


def _norm_matched(model, token_id, layer, alpha):
    """vLLM-Lens's own per-token form: SteeringVector(norm_match=True) adds
    scale * ||h|| * v / ||v|| at every forward, with ||h|| from the summed stream."""
    from vllm_lens import SteeringVector

    vector = SteeringVector(activations=_direction(model, token_id)[None], layer_indices=[layer],
                            scale=alpha, norm_match=True)
    return {"apply_steering_vectors": [vector]}


def _read_last(last):
    def read(ctx, h):
        if ctx.layer_idx == last:
            ctx.saved.setdefault("rows", []).append(_readout(ctx, h))
    return read


@cell("logit_lens", family="llama", backend="vllm_lens")
def logit_lens(be, model, prompts, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]

    def lens(ctx, h):
        ctx.saved.setdefault("rows", []).append(_readout(ctx, h))

    rows = _rows(_hooked(model, prompts[0], lens, idx), len(idx))
    return torch.stack(rows, dim=0)                         # [n_layers, 1, vocab]


@cell("steering", family="llama", backend="vllm_lens")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == "token_norm":
        return _rows(_hooked(model, prompts[0], _read_last(last), [last],
                             extra=_norm_matched(model, token_id, layer, alpha)), 1)[0]
    fn = _steer_fn(layer, token_id, alpha, last)
    return _rows(_hooked(model, prompts[0], fn, [layer, last]), 1)[0]   # [1, vocab]


@cell("gen_steering", family="llama", backend="vllm_lens")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    last = model.n_layers - 1
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == "token_norm":
        saved = _hooked(model, prompts[0], _read_last(last), [last], max_tokens=new_tokens,
                        extra=_norm_matched(model, token_id, layer, alpha))
        return torch.cat(_rows(saved, new_tokens), dim=0)
    fn = _steer_fn(layer, token_id, alpha, last)
    saved = _hooked(model, prompts[0], fn, [layer, last], max_tokens=new_tokens)
    return torch.cat(_rows(saved, new_tokens), dim=0)      # [new_tokens, vocab]


@cell("activation_patching", family="llama", backend="vllm_lens")
def activation_patching(be, model, prompts, *, layer=6, patch=True, positions="all"):
    """Capture the clean residual at `layer` with output_residual_stream, then replace the corrupt
    run's residual there with it and read the corrupt run's next-token logits."""
    from vllm import SamplingParams

    clean_prompt, corrupt_prompt = prompts
    last = model.n_layers - 1
    clean = None
    if patch:
        params = SamplingParams(max_tokens=1, **_GREEDY,
                                extra_args={"output_residual_stream": [layer]})
        (out,) = model.llm.generate([clean_prompt], params, use_tqdm=False)
        clean = out.activations["residual_stream"][0]      # [prompt_len, hidden]

    def transplant(ctx, h):
        if clean is not None and ctx.layer_idx == layer:
            if clean.shape != h.shape:
                raise ValueError(f"patch shape {tuple(clean.shape)} != target {tuple(h.shape)}")
            patched = clean.to(device=h.device, dtype=h.dtype)
            if positions == "last":                        # keep the corrupt rows, swap the last
                patched = torch.cat([h[:-1], patched[-1:]])
            return patched
        if ctx.layer_idx == last:
            ctx.saved.setdefault("rows", []).append(_readout(ctx, h))
        return None

    return _rows(_hooked(model, corrupt_prompt, transplant, [layer, last] if patch else [last]),
                 1)[0]


@cell("ablation", family="llama", backend="vllm_lens")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    if target != "none":
        raise Unsupported(f"hooks fire only at decoder-layer boundaries on the summed residual "
                          f"stream; the {target} output inside a layer is not addressable")
    last = model.n_layers - 1

    def read(ctx, h):
        ctx.saved.setdefault("rows", []).append(_readout(ctx, h))

    return _rows(_hooked(model, prompts[0], read, [last]), 1)[0]

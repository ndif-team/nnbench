"""vLLM-Lens cells in the fastest documented realization (backend `vllm_lens_opt`).

Identical to the system's cells (backends/vllm-lens/cells.py) except where vLLM-Lens 1.2.1 offers
a faster or a documented-but-unused path: the logit lens stacks every layer's last row and reads
them out with one unembedding pass at the last layer (instead of one full-vocab pass per layer),
and ablation is expressed through hooks (a pre-hook's input on vLLM's Qwen2 layers is the previous
layer's MLP output, so zeroing it removes that MLP; the attention output is removed by recomputing
the layer's MLP from the stream without it). A call's units go out as one `generate` over all their
prompts, each request hooked on its own token slice with its own `ctx.saved`, and come back in
prompt order; outputs carry the unit axis at dim -2.

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


def _hooked(model, prompts, fns, layers, *, max_tokens=1, extra=None):
    """One `generate` over `prompts`, request i carrying `fns[i]` as its Hook on `layers`; returns
    each request's saved dict, in prompt order."""
    from vllm import SamplingParams
    from vllm_lens import Hook

    params = [SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens, **_GREEDY, extra_args={
        **(extra or {}), "apply_hooks": [Hook(fn=fn, layer_indices=sorted(set(layers)))]})
        for fn in fns]
    outs = model.llm.generate(list(prompts), params, use_tqdm=False)
    return [out.hook_results["0"] for out in outs]


def _rows(saved, expected):
    rows = saved.get("rows", [])
    if len(rows) != expected:     # a chunked prefill or an early EOS would change the count
        raise RuntimeError(f"collected {len(rows)} readout rows, expected {expected}")
    return rows


def _logits(saved_dicts):
    """Each request's one readout row -> `[B, vocab]`."""
    return torch.cat([_rows(saved, 1)[0] for saved in saved_dicts])


def _steps(saved_dicts, new_tokens):
    """Each request's per-step readout rows -> `[new_tokens, B, vocab]`."""
    return torch.stack([torch.cat(_rows(saved, new_tokens)) for saved in saved_dicts], dim=1)


def _steer_fn(layer, token_id, alpha, last):
    """Add `alpha` times the request's mean per-token residual norm along the target's unembedding
    row at `layer`; read the next-token logits at the last layer. Fires on every forward."""
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
        (saved,) = _hooked(model, [" "], [fetch], [0])
        model.directions[token_id] = F.normalize(saved["row"], dim=0)
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


def _transplant(clean, layer, positions, last):
    """Replace the request's residual at `layer` with its clean snapshot (`None`: no patch), then
    read the next-token logits at the last layer."""
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
    return transplant


@cell("logit_lens", family="llama", backend="vllm_lens_opt")
def logit_lens(be, model, units, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    last = max(idx)

    def lens(ctx, h):
        ctx.saved.setdefault("_rows", []).append(h[-1:])      # a fresh clone per layer, on GPU
        if ctx.layer_idx == last:
            stacked = torch.cat(ctx.saved.pop("_rows"))
            normed = ctx.model.model.norm(stacked)
            ctx.saved["lens"] = F.linear(normed, ctx.get_parameter("lm_head.weight")).float().cpu()

    saved = _hooked(model, units, [lens] * len(units), idx)
    return torch.stack([s["lens"] for s in saved], dim=1)                       # [n_layers, B, vocab]


@cell("steering", family="llama", backend="vllm_lens_opt")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == "token_norm":
        return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last],
                               extra=_norm_matched(model, token_id, layer, alpha)))
    fn = _steer_fn(layer, token_id, alpha, last)
    return _logits(_hooked(model, units, [fn] * len(units), [layer, last]))        # [B, vocab]


@cell("gen_steering", family="llama", backend="vllm_lens_opt")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    last = model.n_layers - 1
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == "token_norm":
        saved = _hooked(model, units, [_read_last(last)] * len(units), [last],
                        max_tokens=new_tokens, extra=_norm_matched(model, token_id, layer, alpha))
    else:
        fn = _steer_fn(layer, token_id, alpha, last)
        saved = _hooked(model, units, [fn] * len(units), [layer, last], max_tokens=new_tokens)
    return _steps(saved, new_tokens)                                            # [new_tokens, B, vocab]


@cell("activation_patching", family="llama", backend="vllm_lens_opt")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all"):
    """Capture every clean residual at `layer` with output_residual_stream, then replace each
    corrupt run's residual there with its pair's and read the corrupt run's next-token logits."""
    from vllm import SamplingParams

    clean_prompts = [unit[0] for unit in units]
    corrupt_prompts = [unit[1] for unit in units]
    last = model.n_layers - 1
    cleans = [None] * len(units)
    if patch:
        params = SamplingParams(max_tokens=1, **_GREEDY,
                                extra_args={"output_residual_stream": [layer]})
        outs = model.llm.generate(clean_prompts, params, use_tqdm=False)
        cleans = [out.activations["residual_stream"][0] for out in outs]      # [prompt_len, hidden]
    fns = [_transplant(clean, layer, positions, last) for clean in cleans]
    return _logits(_hooked(model, corrupt_prompts, fns, [layer, last] if patch else [last]))


def _hooks_request(model, prompts, hooks, *, read_index):
    """One `generate` over `prompts` with the same several hooks on every request (each request
    keeps its own contexts); returns each request's saved dict of hook `read_index`."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, **_GREEDY, extra_args={"apply_hooks": hooks})
    outs = model.llm.generate(list(prompts), params, use_tqdm=False)
    return [out.hook_results[str(read_index)] for out in outs]


@cell("ablation", family="llama", backend="vllm_lens_opt")
def ablation(be, model, units, *, layer=6, target="mlp"):
    from vllm_lens import Hook

    last = model.n_layers - 1
    readout = Hook(fn=_read_last(last), layer_indices=[last])
    if target == "none":
        return _logits(_hooks_request(model, units, [readout], read_index=0))
    if target == "mlp":
        if layer + 1 > last:
            raise Unsupported("zeroing the last layer's MLP needs a pre-hook on the layer after it")

        def zero_mlp(ctx, h):
            return torch.zeros_like(h)          # layer+1's hidden input is layer's MLP output

        hooks = [Hook(fn=zero_mlp, layer_indices=[layer + 1], pre=True), readout]
        return _logits(_hooks_request(model, units, hooks, read_index=1))
    if target == "attn":
        if layer == 0:
            raise Unsupported("the stream before layer 0 is not a decoder-layer output")

        def drop_attn(ctx, h):
            if ctx.layer_idx == layer - 1:
                ctx.saved["_before"] = h                 # the stream entering `layer`
                return None
            before = ctx.saved.pop("_before")
            block = ctx.model.model.layers[layer]
            return before + block.mlp(block.post_attention_layernorm(before))

        hooks = [Hook(fn=drop_attn, layer_indices=[layer - 1, layer]), readout]
        return _logits(_hooks_request(model, units, hooks, read_index=1))
    raise ValueError(f"unknown ablation target {target!r}")

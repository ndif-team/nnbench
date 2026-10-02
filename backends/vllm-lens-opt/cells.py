"""vLLM-Lens cells in the fastest documented realization (backend `vllm_lens_opt`).

Identical to backends/vllm-lens/cells.py except where vLLM-Lens 1.2.1 offers a faster or a
documented-but-unused path: the logit lens stacks every layer's last row and reads them out with
one unembedding pass at the last layer (instead of one full-vocab pass per layer), and ablation is
expressed through hooks (a pre-hook's input on vLLM's Qwen2 layers is the previous layer's MLP
output, so zeroing it removes that MLP; the attention output is removed by recomputing the
layer's MLP from the stream without it).

vLLM-Lens cells for the comparison specs (design.md §12.14).

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


@cell("logit_lens", family="llama", backend="vllm_lens_opt")
def logit_lens(be, model, prompts, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    last = max(idx)

    def lens(ctx, h):
        ctx.saved.setdefault("_rows", []).append(h[-1:])      # a fresh clone per layer, on GPU
        if ctx.layer_idx == last:
            stacked = torch.cat(ctx.saved.pop("_rows"))
            normed = ctx.model.model.norm(stacked)
            ctx.saved["lens"] = F.linear(normed, ctx.get_parameter("lm_head.weight")).float().cpu()

    saved = _hooked(model, prompts[0], lens, idx)
    return saved["lens"].unsqueeze(1)                          # [n_layers, 1, vocab]


@cell("steering", family="llama", backend="vllm_lens_opt")
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


@cell("gen_steering", family="llama", backend="vllm_lens_opt")
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


@cell("activation_patching", family="llama", backend="vllm_lens_opt")
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


def _hooks_request(model, prompt, hooks, *, read_index):
    """One request with several hooks; returns the saved dict of hook `read_index`."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, **_GREEDY, extra_args={"apply_hooks": hooks})
    (out,) = model.llm.generate([prompt], params, use_tqdm=False)
    return out.hook_results[str(read_index)]


@cell("ablation", family="llama", backend="vllm_lens_opt")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    from vllm_lens import Hook

    last = model.n_layers - 1
    readout = Hook(fn=_read_last(last), layer_indices=[last])
    if target == "none":
        return _rows(_hooks_request(model, prompts[0], [readout], read_index=0), 1)[0]
    if target == "mlp":
        if layer + 1 > last:
            raise Unsupported("zeroing the last layer's MLP needs a pre-hook on the layer after it")

        def zero_mlp(ctx, h):
            return torch.zeros_like(h)          # layer+1's hidden input is layer's MLP output

        hooks = [Hook(fn=zero_mlp, layer_indices=[layer + 1], pre=True), readout]
        return _rows(_hooks_request(model, prompts[0], hooks, read_index=1), 1)[0]
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
        return _rows(_hooks_request(model, prompts[0], hooks, read_index=1), 1)[0]
    raise ValueError(f"unknown ablation target {target!r}")

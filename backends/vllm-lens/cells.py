"""vLLM-Lens cells for the cross-system workloads.

Each cell expresses the workload row's semantics with vLLM-Lens 1.2.1's own mechanisms: `Hook`
functions on decoder layers (post-hooks see the summed residual stream, and a returned tensor
replaces it) and `output_residual_stream` capture. A call's units go out as one `generate` over all
their prompts, each request carrying its own hook or capture: the worker runs a request's hook on
that request's token slice with that request's own `ctx.saved` (`_worker_ext.py`, `_hook_inner`:
per-request `query_start_loc` slices and `store[req_id]` contexts) and returns `hook_results` and
`activations` per request, in prompt order. Readouts run on the worker inside a hook, as in
vLLM-Lens's logit-lens example: the model's final norm, then the unembedding weight from
`ctx.get_parameter`. Outputs carry the unit axis at dim -2.

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


@cell("logit_lens", family="llama", backend="vllm_lens")
def logit_lens(be, model, units, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]

    def lens(ctx, h):
        ctx.saved.setdefault("rows", []).append(_readout(ctx, h))

    saved = _hooked(model, units, [lens] * len(units), idx)
    return torch.stack([torch.cat(_rows(s, len(idx))) for s in saved], dim=1)   # [n_layers, B, vocab]


@cell("steering", family="llama", backend="vllm_lens")
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


@cell("gen_steering", family="llama", backend="vllm_lens")
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


@cell("activation_patching", family="llama", backend="vllm_lens")
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


@cell("ablation", family="llama", backend="vllm_lens")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target != "none":
        raise Unsupported(f"hooks fire only at decoder-layer boundaries on the summed residual "
                          f"stream; the {target} output inside a layer is not addressable")
    last = model.n_layers - 1
    return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last]))


def _captured(model, prompts, layer, max_tokens, export):
    """One generate call for `prompts`; with `export`, vLLM-Lens's activation capture at `layer`."""
    from vllm import SamplingParams

    extra = {"output_residual_stream": [layer]} if export else None
    params = SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens, **_GREEDY, extra_args=extra)
    return model.llm.generate(list(prompts), params, use_tqdm=False)


@cell("activation_harvest", family="llama", backend="vllm_lens")
def activation_harvest(be, model, units, *, layer, export=True):
    """The documented activation capture, `output_residual_stream`, on one generate call for the
    whole batch. At every forward the worker copies each request's slice of the summed stream
    (hidden + residual after the layer) to host memory (`_worker_ext.py`, `_hook_inner` phase 3),
    and the plugin returns it as `[1, n_prompt, d]` per request at max_tokens=1
    (`_activations_plugin.py`, `_finalize_offline_outputs`). vLLM's `LLM.generate` returns outputs
    in prompt order. `export=False` runs the same call without capture and returns the greedy
    tokens."""
    outs = _captured(model, units, layer, 1, export)
    if not export:
        return torch.tensor([[out.outputs[0].token_ids[0]] for out in outs])      # [B, 1]
    return torch.cat([out.activations["residual_stream"][0] for out in outs])   # [tokens, d]


@cell("gen_capture", family="llama", backend="vllm_lens")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    """The same capture over a greedy decode: it appends every decode step's row and is trimmed to
    n_prompt + n_gen - 1 positions (`_finalize_offline_outputs`). The rows from the last prompt
    token on are the newest position of each step, the rows the nnsight cell reads."""
    outs = _captured(model, units, layer, new_tokens, export)
    rows = []
    for out in outs:
        steps = (out.activations["residual_stream"][0][len(out.prompt_token_ids) - 1:] if export
                 else torch.tensor(out.outputs[0].token_ids)[:, None])
        if len(steps) != new_tokens:     # an early EOS shortens the decode
            raise RuntimeError(f"collected {len(steps)} per-step rows, expected {new_tokens}")
        rows.append(steps)
    return torch.stack(rows, dim=1)                              # [new_tokens, B, d] or [.., B, 1]

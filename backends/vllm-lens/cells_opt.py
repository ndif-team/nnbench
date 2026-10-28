"""vLLM-Lens cells in the fastest documented realization (backend `vllm_lens_opt`).

Identical to the system's cells (backends/vllm-lens/cells.py) except where vLLM-Lens 1.2.1 offers
a faster or a documented-but-unused path: the logit lens stacks every layer's last row and reads
them out with one unembedding pass at the last layer (instead of one full-vocab pass per layer),
and ablation is expressed through hooks (a pre-hook's input on vLLM's Qwen2 layers is the previous
layer's MLP output, so zeroing it removes that MLP; the attention output is removed by recomputing
the layer's MLP from the stream without it). A call's units go out as one `generate` over all their
prompts, each request hooked on its own token slice with its own `ctx.saved`, and come back in
prompt order; outputs carry the unit axis at dim -2. Readouts are computed and kept on
tensor-parallel rank 0 only (`_rank0`), as in cells.py.

Registered for family "llama" (vLLM module paths `model.norm`, `lm_head`; vLLM-Lens itself finds
decoder layers only under `model.layers`-style trees).
"""
import functools
import sys

import cloudpickle
import torch
import torch.nn.functional as F

from isb.methodologies.jacobian_lens import _load_fitted, _orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import CALIBRATED, _resolve_token, calibrated_strengths
from isb.tasks.lens_eval import locate

# The hook closures travel to the engine-core process by value; this module is not on its path.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

_GREEDY = {"temperature": 0.0, "top_p": 1.0}


def _rank0():
    """Whether this worker is tensor-parallel rank 0, the one rank that computes and keeps a
    request's readouts. Every TP rank runs each hook on the same replicated stream, and the plugin
    merges the ranks' `ctx.saved` dicts (`_activations_plugin.py`, `_merge_hook_results`: list
    values concatenated, any other value overwritten by a later rank's, a key a rank lacks left
    as the others set it), so a value kept on rank 0 alone arrives once and is computed once.
    Under TP the backend prefetches `lm_head.weight`, so `ctx.get_parameter` on rank 0 alone runs
    no collective (`_helpers/types.py`, `HookContext.get_parameter`)."""
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank() == 0


def _append(ctx, key, value):
    """Append to a per-request list in `ctx.saved` (callers keep readouts on `_rank0` only)."""
    ctx.saved.setdefault(key, []).append(value)


def _unembed(ctx, x):
    """Logits of the rows `x`: final norm, then the unembedding, in the engine's dtype on the GPU."""
    return F.linear(ctx.model.model.norm(x), ctx.get_parameter("lm_head.weight"))


def _readout(ctx, h):
    """Next-token logits of the last row: final norm, then the unembedding. Runs on the worker."""
    return _unembed(ctx, h[-1:]).float().cpu()


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


def _steer_fn(layers, token_id, alpha, last):
    """Add `alpha` times the request's mean per-token residual norm along the target's unembedding
    row at each of `layers` (each layer's norm from the stream it sees, earlier writes included);
    read the next-token logits at the last layer. Fires on every forward; the direction is built
    once per request, at the first steered layer."""
    layers = frozenset(layers)
    built = []

    def steer(ctx, h):
        if ctx.layer_idx in layers and alpha != 0:
            if not built:
                weight = ctx.get_parameter("lm_head.weight")
                built.append(F.normalize(weight[token_id].float(), dim=0).to(weight.dtype))
            return h + (alpha * h.norm(dim=-1).mean()) * built[0]
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _readout(ctx, h))
        return None
    return steer


def _direction(model, token_id):
    """The target token's unit unembedding row, fetched once from the worker through a hook."""
    if token_id not in model.directions:
        def fetch(ctx, h):
            if _rank0():
                ctx.saved["row"] = ctx.get_parameter("lm_head.weight")[token_id].float().cpu()
        (saved,) = _hooked(model, [" "], [fetch], [0])
        model.directions[token_id] = F.normalize(saved["row"], dim=0)
    return model.directions[token_id]


def _norm_matched(model, token_id, layers, alpha):
    """vLLM-Lens's own per-token form: SteeringVector(norm_match=True) adds
    scale * ||h|| * v / ||v|| at every forward of each of `layers`, with ||h|| from the summed
    stream."""
    from vllm_lens import SteeringVector

    direction = _direction(model, token_id)
    vector = SteeringVector(activations=direction.expand(len(layers), -1).contiguous(),
                            layer_indices=list(layers), scale=alpha, norm_match=True)
    return {"apply_steering_vectors": [vector]}


def _mean_norms(model, prompts, layers):
    """The unsteered summed stream's mean per-token norm at each of `layers`, averaged over
    `prompts`: one hooked `generate` whose hook only reads (`calibrated_strengths`)."""
    def read(ctx, h):
        if _rank0():
            ctx.saved[f"norm{ctx.layer_idx}"] = h.float().norm(dim=-1).mean().item()

    saved = _hooked(model, prompts, [read] * len(prompts), layers)
    return [sum(s[f"norm{i}"] for s in saved) / len(saved) for i in layers]


def _fixed_vectors(model, token_id, layers, strengths):
    """vLLM-Lens's documented steering form with fixed vectors (README "Steering vectors"):
    SteeringVector adds `scale * activations[i]` to the summed stream at `layer_indices[i]` on every
    forward (`_worker_ext._apply_steering`, norm_match off); row i is `strengths[i]` times the
    target's unit unembedding row."""
    from vllm_lens import SteeringVector

    direction = _direction(model, token_id)
    vector = SteeringVector(activations=torch.stack([s * direction for s in strengths]),
                            layer_indices=list(layers), scale=1.0)
    return {"apply_steering_vectors": [vector]}


def _read_last(last):
    def read(ctx, h):
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _readout(ctx, h))
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
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _readout(ctx, h))
        return None
    return transplant


@cell("logit_lens", family="llama", backend="vllm_lens_opt")
def logit_lens(be, model, units, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    last = max(idx)

    def lens(ctx, h):
        if not _rank0():
            return
        ctx.saved.setdefault("_rows", []).append(h[-1:])      # a fresh clone per layer, on GPU
        if ctx.layer_idx == last:
            stacked = torch.cat(ctx.saved.pop("_rows"))
            normed = ctx.model.model.norm(stacked)
            ctx.saved["lens"] = F.linear(normed, ctx.get_parameter("lm_head.weight")).float().cpu()

    saved = _hooked(model, units, [lens] * len(units), idx)
    return torch.stack([s["lens"] for s in saved], dim=1)                       # [n_layers, B, vocab]


@cell("steering", family="llama", backend="vllm_lens_opt")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm"):
    """`layers` steers every listed layer in the one forward (and `layer` is ignored), each with
    `alpha / len(layers)`. `scale_by="calibrated_mean_norm"` adds fixed vectors through
    `apply_steering_vectors`, their norms measured once by a read-only hooked request
    (isb/methodologies/steering.py, `calibrated_strengths`)."""
    last = model.n_layers - 1
    steered = [i % model.n_layers for i in (layers if layers is not None else [layer])]
    if alpha != 0 and last in steered:
        raise ValueError(f"steer layers {steered} include the read-out layer; pick layers < {last}")
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == CALIBRATED:
        extra = None
        if alpha != 0:
            strengths = calibrated_strengths(be, steered, alpha, units,
                                             functools.partial(_mean_norms, model))
            extra = _fixed_vectors(model, token_id, steered, strengths)
        return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last],
                               extra=extra))                                     # [B, vocab]
    share = alpha / len(steered)
    if scale_by == "token_norm":
        return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last],
                               extra=_norm_matched(model, token_id, steered, share)))
    fn = _steer_fn(steered, token_id, share, last)
    return _logits(_hooked(model, units, [fn] * len(units), [*steered, last]))     # [B, vocab]


@cell("gen_steering", family="llama", backend="vllm_lens_opt")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    last = model.n_layers - 1
    token_id = _resolve_token(model.tokenizer, target)
    if scale_by == "token_norm":
        saved = _hooked(model, units, [_read_last(last)] * len(units), [last],
                        max_tokens=new_tokens, extra=_norm_matched(model, token_id, [layer], alpha))
    else:
        fn = _steer_fn([layer], token_id, alpha, last)
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


def _prefill_transplant(clean, layer, last):
    """Replace the request's residual at `layer` with its clean snapshot on the request's first
    forward only (the prefill, which holds the whole prompt), then read the next-token logits at the
    last layer on every forward. The returned stream is applied as a delta onto the layer's first
    element (`_worker_ext.py`, `_apply_hook_delta`), so the summed stream after the layer becomes
    the clean one at every prompt position, and the decode steps carry it through the request's
    own KV cache. `clean=None` reads only. The prefill's shape goes to `ctx.saved["patched"]`: a
    hook that raises is logged and skipped by the plugin (`_make_hook`), so the client checks it."""
    def transplant(ctx, h):
        out = None
        if clean is not None and ctx.layer_idx == layer and "patched" not in ctx.saved:
            ctx.saved["patched"] = tuple(h.shape)
            if tuple(clean.shape) == tuple(h.shape):
                out = clean.to(device=h.device, dtype=h.dtype)
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _unembed(ctx, (h if out is None else out)[-1:]).cpu())
        return out
    return transplant


@cell("gen_patching", family="llama", backend="vllm_lens_opt")
def gen_patching(be, model, units, *, layer=9, new_tokens=8, patch=True):
    """As in cells.py: vLLM-Lens 1.2.1 has no faster documented form of a prefill transplant
    under decode. Capture every clean residual at `layer` with output_residual_stream, then
    greedily decode each corrupt prompt `new_tokens` steps with its pair's clean residual
    transplanted at the prefill, reading each step's next-token logits: `[new_tokens, B, vocab]`
    in the engine's dtype. `patch=False` is the same decode without the transplant."""
    from vllm import SamplingParams

    corrupt_prompts = [unit[1] for unit in units]
    last = model.n_layers - 1
    cleans = [None] * len(units)
    if patch:
        params = SamplingParams(max_tokens=1, **_GREEDY,
                                extra_args={"output_residual_stream": [layer]})
        outs = model.llm.generate([unit[0] for unit in units], params, use_tqdm=False)
        cleans = [out.activations["residual_stream"][0] for out in outs]      # [prompt_len, d]
    fns = [_prefill_transplant(clean, layer, last) for clean in cleans]
    saved = _hooked(model, corrupt_prompts, fns, [layer, last] if patch else [last],
                    max_tokens=new_tokens)
    for clean, s in zip(cleans, saved):
        if clean is not None and s.get("patched") != tuple(clean.shape):
            raise ValueError(f"transplant shape {tuple(clean.shape)} != prefill residual "
                             f"{s.get('patched')}; clean/corrupted prompts must be length-matched "
                             f"and the prefill unchunked for position-aligned patching")
    return _steps(saved, new_tokens)                                            # [new_tokens, B, vocab]


def _transport(model, transport, layers):
    """The J-lens transport for the read `layers` in the engine's dtype on the host: None (the
    identity), one `[d, d]` map shared by every layer (an int seed, `_orthogonal`, or a tensor), or
    an `[n_layers, d, d]` stack of per-layer maps (a fitted lens, `_load_fitted`, or a mapping).
    A seed or a fitted spec is built once per model and kept."""
    if transport is None:
        return None
    key = (transport, tuple(layers)) if isinstance(transport, (int, str)) else None
    if key in model.transports:
        return model.transports[key]
    if isinstance(transport, int):
        maps = _orthogonal(model.config.hidden_size, transport)
    elif isinstance(transport, torch.Tensor):
        maps = transport
    else:
        per_layer = _load_fitted(transport) if isinstance(transport, str) else transport
        maps = torch.stack([per_layer[i] for i in layers])
    maps = maps.to(model.dtype)
    if key is not None:
        model.transports[key] = maps
    return maps


@cell("jacobian_lens", family="llama", backend="vllm_lens_opt")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """The J-lens readout stacked like the opt logit lens: a Hook keeps the request's row at
    `position` (an index into its own prompt tokens) from the summed stream of every read layer
    on the GPU, and at the deepest one applies the transport (one matmul for a shared map, one
    batched matmul for per-layer maps), the final norm and the unembedding in one pass, then
    copies `[n_layers, vocab]` to the host. The maps travel with every request in the hook
    closure, as in vLLM-Lens's jacobian-lens example. Output `[n_layers, B, vocab]` in the
    engine's dtype."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    hooked = sorted(set(idx))                         # the order the hook fires and stacks in
    maps = _transport(model, transport, hooked)
    last = hooked[-1]

    def lens_at(position):
        def lens(ctx, h):
            if not _rank0():
                return
            ctx.saved.setdefault("_rows", []).append(h[position].unsqueeze(0))   # on GPU
            if ctx.layer_idx == last:
                x = torch.cat(ctx.saved.pop("_rows"))                           # [n_layers, d]
                if maps is not None:
                    J = maps.to(h.device)
                    x = x @ J.T if J.dim() == 2 else torch.bmm(J, x.unsqueeze(-1)).squeeze(-1)
                if "lens" in ctx.saved:                  # a chunked prefill reached the last layer twice
                    ctx.saved["chunked"] = True
                ctx.saved["lens"] = _unembed(ctx, x).cpu()
        return lens

    fns = [lens_at(locate(model.tokenizer, unit, position)) for unit in units]
    saved = _hooked(model, units, fns, hooked)
    if any(s.get("chunked") for s in saved):
        raise RuntimeError("a prefill was chunked; the readout position is not in one forward")
    stacked = torch.stack([s["lens"] for s in saved], dim=1)                    # [n_hooked, B, vocab]
    return stacked[[hooked.index(i) for i in idx]]                              # [n_layers, B, vocab]


@cell("attention_pattern", family="llama", backend="vllm_lens_opt")
def attention_pattern(be, model, units, *, layers="all"):
    raise Unsupported("vLLM-Lens 1.2.1 hooks only decoder layers (`_worker_ext.py`, "
                      "`install_hooks`: register_forward_pre_hook/register_forward_hook on each "
                      "layer of `_get_layers`) and hands a hook the layer's hidden-state slice; it "
                      "has no capture of the attention module's query and key (Q/K capture, "
                      "`output_qk` and `vllm_lens/attention.py`, arrives in 1.3.0)")


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


from .cells import activation_harvest, gen_capture, residual_readout  # noqa: E402

cell("residual_readout", family="llama", backend="vllm_lens_opt")(residual_readout)
cell("activation_harvest", family="llama", backend="vllm_lens_opt")(activation_harvest)
cell("gen_capture", family="llama", backend="vllm_lens_opt")(gen_capture)

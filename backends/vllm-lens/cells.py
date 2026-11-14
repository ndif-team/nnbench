"""vLLM-Lens cells for the cross-system workloads.

Each cell expresses the workload row's semantics with vLLM-Lens's own mechanisms, the same in
1.2.1 and 1.3.0: `Hook` functions on decoder layers (post-hooks see the summed residual stream,
and a returned tensor replaces it) and `output_residual_stream` capture. A call's units go out as
one `generate` over all their prompts, each request carrying its own hook or capture: the worker
runs a request's hook on that request's token slice with that request's own `ctx.saved`
(`_worker_ext.py`, `_hook_inner`: per-request `query_start_loc` slices and `store[req_id]`
contexts) and returns `hook_results` and `activations` per request, in prompt order. Readouts run
on the worker inside a hook, as in vLLM-Lens's logit-lens example: the model's final norm, then
the unembedding weight from `ctx.get_parameter`. A timed call (`check=False`) runs only the
methodology: an intervention sends no readout and returns the tokens its requests generated
(`RequestOutput.outputs[0].token_ids`), a read returns its rows in the dtype the worker hands
back; a check call adds the readouts, fp32 casts and guards that scoring needs. Outputs carry the
unit axis at dim -2, attention patterns at dim -3 (before heads and keys). Under tensor
parallelism every rank runs each hook on the replicated stream and the plugin merges the ranks'
`ctx.saved` dicts, so readouts are computed and kept on rank 0 only (`_rank0`).

Registered for family "llama" (vLLM module paths `model.norm`, `lm_head`; vLLM-Lens 1.2.1 finds
decoder layers only under `model.layers`-style trees, `_get_layers`, and 1.3.0 from the attention
modules vLLM registers, `_discover_layer_modules`).

1.3.0 adds a capture of the attention kernel's query and key (`output_qk`), which the
attention-pattern cell reads when that release is installed (`_QK`).
"""
import functools
import sys
from importlib import metadata

import cloudpickle
import torch
import torch.nn.functional as F

from isb.methodologies.jacobian_lens import load_fitted, orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import CALIBRATED, resolve_token, calibrated_strengths
from isb.data import locate

# The hook closures travel to the engine-core process by value; this module is not on its path.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

_GREEDY = {"temperature": 0.0, "top_p": 1.0}

# Whether the installed release captures the attention kernel's query and key (`output_qk`).
_QK = tuple(map(int, metadata.version("vllm-lens").split(".")[:2])) >= (1, 3)


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
    each request's saved dict, in prompt order. Requests that carry the same function share one
    SamplingParams, as a user's batch does: the plugin ships one hook payload per SamplingParams
    and keeps each request's results apart (`_hook_contexts`, keyed by request)."""
    from vllm import SamplingParams
    from vllm_lens import Hook

    def params(fn):
        return SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens, **_GREEDY, extra_args={
            **(extra or {}), "apply_hooks": [Hook(fn=fn, layer_indices=sorted(set(layers)))]})

    sampling = params(fns[0]) if all(fn is fns[0] for fn in fns) else [params(fn) for fn in fns]
    outs = model.llm.generate(list(prompts), sampling, use_tqdm=False)
    return [out.hook_results["0"] for out in outs]


def _rows(saved, expected=None):
    """A request's readout rows; with `expected`, the check call's guard that it collected that
    many (a chunked prefill or an early EOS would change the count)."""
    rows = saved.get("rows", [])
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"collected {len(rows)} readout rows, expected {expected}")
    return rows


def _logits(saved_dicts):
    """Each request's one readout row -> `[B, vocab]`."""
    return torch.cat([_rows(saved, 1)[0] for saved in saved_dicts])


def _steps(saved_dicts, new_tokens):
    """Each request's per-step readout rows -> `[new_tokens, B, vocab]`."""
    return torch.stack([torch.cat(_rows(saved, new_tokens)) for saved in saved_dicts], dim=1)


def _in_order(stacked, hooked, listed):
    """`stacked`, one entry per layer of `hooked` (ascending) along dim 0, in the `listed` order:
    `stacked` itself when the two orders agree, else a reindexed copy."""
    return stacked if listed == hooked else stacked[[hooked.index(i) for i in listed]]


def _hook_args(fns, layers):
    """One extra_args per request, request i carrying `fns[i]` as its Hook on `layers`."""
    from vllm_lens import Hook

    return [{"apply_hooks": [Hook(fn=fn, layer_indices=sorted(set(layers)))]} for fn in fns]


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


def mean_norms(model, prompts, layers):
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
    """Replace the request's residual at `layer` with its clean snapshot (`None`: no patch), and
    read the next-token logits at `last` (`None`: no read)."""
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


@cell("logit_lens", family="llama", backend="vllm_lens")
def logit_lens(be, model, units, *, layers="all", check=False):
    """A Hook on every read layer projects the request's last row through the final norm and the
    unembedding on the worker and copies it to the host, as in vLLM-Lens's logit-lens example:
    `[n_layers, B, vocab]`, in the engine's dtype for a timed call and in fp32 for a check call,
    which also checks each request collected one row per layer."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]

    def lens(ctx, h):
        if _rank0():
            _append(ctx, "rows", _readout(ctx, h) if check else _unembed(ctx, h[-1:]).cpu())

    saved = _hooked(model, units, [lens] * len(units), idx)
    expected = len(idx) if check else None
    return torch.stack([torch.cat(_rows(s, expected)) for s in saved], dim=1)   # [n_layers, B, vocab]


def _generated_tokens(model, prompts, extra=None, *, new_tokens=1):
    """Each prompt's `new_tokens` greedy tokens among the non-stop tokens (`min_tokens`, as the
    plain request), `[B, new_tokens]` int64, read off vLLM's request outputs in prompt order
    (`outputs[0].token_ids`), as the README runs a steered or hooked generation. `extra` is the
    extra_args every request carries, a list of one per prompt, or None (the plain request); each
    dict is copied, since the plugin pops the steer and the hooks off it
    (`vllm_lens/_activations_plugin.py`, `_prepare_offline_params`)."""
    from vllm import SamplingParams

    def params(args):
        return SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, **_GREEDY,
                              extra_args=dict(args) if args else None)

    sampling = [params(args) for args in extra] if isinstance(extra, list) else params(extra)
    outs = model.llm.generate(list(prompts), sampling, use_tqdm=False)
    return torch.tensor([list(out.outputs[0].token_ids) for out in outs])


def _mean_norm_fn(layers, direction, share, last):
    """The mean-norm steer as a Hook function: at each of `layers` add `share` times the request's
    mean per-token norm of the stream that layer sees (earlier writes included) along `direction`,
    which moves onto the stream's device and dtype at the first steered layer; at `last` read the
    next-token logits. Registered on `layers` alone it only steers; on `layers` and `last` it also
    reads."""
    layers = frozenset(layers)
    moved = []

    def steer(ctx, h):
        if ctx.layer_idx in layers:
            if not moved:
                moved.append(direction.to(device=h.device, dtype=h.dtype))
            return h + (share * h.norm(dim=-1).mean()) * moved[0]
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _readout(ctx, h))
        return None
    return steer


def _steer_args(be, model, units, steered, target, alpha, scale_by):
    """The extra_args that steer a request toward `target` at the `steered` layers, built once per
    loaded engine and kept on `model.steers`, so warm-up calls build them and the timed and check
    calls carry the same objects. "calibrated_mean_norm" and "token_norm" are SteeringVectors
    (`_fixed_vectors`, `_norm_matched`); the mean norm, which the per-token norm_match of a
    SteeringVector cannot express (`vllm_lens/_worker_ext.py`, `norm_match`), is a Hook on
    `_mean_norm_fn`."""
    key = (scale_by, target, tuple(steered), alpha)
    if key not in model.steers:
        from vllm_lens import Hook

        token_id = resolve_token(model.tokenizer, target)
        share = alpha / len(steered)
        if scale_by == CALIBRATED:
            strengths = calibrated_strengths(be, steered, alpha, units,
                                             functools.partial(mean_norms, model))
            model.steers[key] = _fixed_vectors(model, token_id, steered, strengths)
        elif scale_by == "token_norm":
            model.steers[key] = _norm_matched(model, token_id, steered, share)
        else:
            fn = _mean_norm_fn(steered, _direction(model, token_id), share, model.n_layers - 1)
            model.steers[key] = {"apply_hooks": [Hook(fn=fn, layer_indices=sorted(set(steered)))]}
    return model.steers[key]


@cell("steering", family="llama", backend="vllm_lens")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm", check=False):
    """`layers` steers every listed layer in the one forward (and `layer` is ignored), each with
    `alpha / len(layers)`; `scale_by="calibrated_mean_norm"` adds fixed vectors whose norms one
    read-only hooked request measures (isb/methodologies/steering.py, `calibrated_strengths`). A
    timed call sends the steer (`_steer_args`), or at `alpha=0` the plain request, on one greedy
    one-token `generate` and returns each request's generated token, `[B, 1]`. A check call
    carries the same steer and reads the next-token logits on the worker at the last layer,
    `[B, vocab]`."""
    last = model.n_layers - 1
    steered = [i % model.n_layers for i in (layers if layers is not None else [layer])]
    if alpha != 0 and last in steered:
        raise ValueError(f"steer layers {steered} include the read-out layer; pick layers < {last}")
    steer = _steer_args(be, model, units, steered, target, alpha, scale_by) if alpha != 0 else None
    if not check:
        return _generated_tokens(model, units, steer)                              # [B, 1]
    if steer is not None and "apply_hooks" in steer:      # the mean-norm Hook, on `last` too
        (hook,) = steer["apply_hooks"]
        return _logits(_hooked(model, units, [hook.fn] * len(units), [*steered, last]))
    return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last],
                           extra=steer))                                           # [B, vocab]


@cell("gen_steering", family="llama", backend="vllm_lens")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm", check=False):
    """The steering cell's steer at `layer` (`_steer_args`: each token's norm for "token_norm", the
    mean norm for any other scale) on every forward of a greedy `new_tokens`-step decode, or at
    `alpha=0` the plain request. A timed call returns the generated tokens, `[new_tokens, B, 1]`; a
    check call carries the same steer and reads each step's next-token logits at the last layer,
    `[new_tokens, B, vocab]`."""
    last = model.n_layers - 1
    scale = "token_norm" if scale_by == "token_norm" else "mean_norm"
    steer = _steer_args(be, model, units, [layer], target, alpha, scale) if alpha != 0 else None
    if not check:
        return _generated_tokens(model, units, steer, new_tokens=new_tokens).t().unsqueeze(-1)
    if steer is not None and "apply_hooks" in steer:      # the mean-norm Hook, on `last` too
        (hook,) = steer["apply_hooks"]
        saved = _hooked(model, units, [hook.fn] * len(units), [layer, last], max_tokens=new_tokens)
    else:
        saved = _hooked(model, units, [_read_last(last)] * len(units), [last],
                        max_tokens=new_tokens, extra=steer)
    return _steps(saved, new_tokens)                                            # [new_tokens, B, vocab]


@cell("activation_patching", family="llama", backend="vllm_lens")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all", check=False):
    """Capture every clean residual at `layer` with output_residual_stream, then replace each
    corrupt run's residual there with its pair's (`patch=False`: the corrupt runs alone). A timed
    call sends the transplant alone and returns each corrupt request's generated token, `[B, 1]`;
    a check call also reads the corrupt run's next-token logits at the last layer, `[B, vocab]`."""
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
    if not check:
        patches = (_hook_args([_transplant(clean, layer, positions, None) for clean in cleans],
                              [layer]) if patch else None)
        return _generated_tokens(model, corrupt_prompts, patches)                  # [B, 1]
    fns = [_transplant(clean, layer, positions, last) for clean in cleans]
    return _logits(_hooked(model, corrupt_prompts, fns, [layer, last] if patch else [last]))


@cell("ablation", family="llama", backend="vllm_lens")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """The baseline alone (`target="none"`): a timed call is the plain request and returns each
    request's generated token, `[B, 1]`; a check call reads the next-token logits at the last layer,
    `[B, vocab]`."""
    if target != "none":
        raise Unsupported(f"hooks fire only at decoder-layer boundaries on the summed residual "
                          f"stream; the {target} output inside a layer is not addressable")
    if not check:
        return _generated_tokens(model, units)                                      # [B, 1]
    last = model.n_layers - 1
    return _logits(_hooked(model, units, [_read_last(last)] * len(units), [last]))


def _prefill_transplant(clean, layer, last):
    """Replace the request's residual at `layer` with its clean snapshot on the request's first
    forward only (the prefill, which holds the whole prompt), and read the next-token logits at
    `last` (`None`: no read) on every forward. The returned stream is written into the layer's
    output (1.2.1 adds the difference onto the first element, `_worker_ext.py`,
    `_apply_hook_delta`; 1.3.0 stores it there and zeroes the second, `_helpers/_hook_output.py`),
    so the summed stream after the layer becomes the clean one at every prompt position, and the
    decode steps carry it through the request's own KV cache. `clean=None` reads only. The
    prefill's shape goes to `ctx.saved["patched"]`: a hook that raises is logged and skipped by the
    plugin (`_make_hook`), so the client checks it."""
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


@cell("gen_patching", family="llama", backend="vllm_lens")
def gen_patching(be, model, units, *, layer=9, new_tokens=8, patch=True, check=False):
    """Capture every clean residual at `layer` with output_residual_stream (`[1, n_prompt, d]` of
    the summed stream per request), then greedily decode each corrupt prompt `new_tokens` steps
    with its pair's clean residual transplanted at the prefill (`patch=False`: without it). A timed
    call returns the generated tokens, `[new_tokens, B, 1]`; a check call reads each step's
    next-token logits, `[new_tokens, B, vocab]` in the engine's dtype, and checks every prefill
    took the transplant."""
    from vllm import SamplingParams

    corrupt_prompts = [unit[1] for unit in units]
    last = model.n_layers - 1
    cleans = [None] * len(units)
    if patch:
        params = SamplingParams(max_tokens=1, **_GREEDY,
                                extra_args={"output_residual_stream": [layer]})
        outs = model.llm.generate([unit[0] for unit in units], params, use_tqdm=False)
        cleans = [out.activations["residual_stream"][0] for out in outs]      # [prompt_len, d]
    if not check:
        patches = (_hook_args([_prefill_transplant(clean, layer, None) for clean in cleans],
                              [layer]) if patch else None)
        tokens = _generated_tokens(model, corrupt_prompts, patches, new_tokens=new_tokens)
        return tokens.t().unsqueeze(-1)                                         # [new_tokens, B, 1]
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
    identity), one `[d, d]` map shared by every layer (an int seed, `orthogonal`, or a tensor), or
    an `[n_layers, d, d]` stack of per-layer maps (a fitted lens, `load_fitted`, or a mapping).
    A seed or a fitted spec is built once per model and kept."""
    if transport is None:
        return None
    key = (transport, tuple(layers)) if isinstance(transport, (int, str)) else None
    if key in model.transports:
        return model.transports[key]
    if isinstance(transport, int):
        maps = orthogonal(model.config.hidden_size, transport)
    elif isinstance(transport, torch.Tensor):
        maps = transport
    else:
        per_layer = load_fitted(transport) if isinstance(transport, str) else transport
        maps = torch.stack([per_layer[i] for i in layers])
    maps = maps.to(model.dtype)
    if key is not None:
        model.transports[key] = maps
    return maps


def _jlens_fn(maps, layers, position):
    """The J-lens readout of the request's row at `position` (an index into its own prompt tokens,
    the prefill's slice): `J_l @ h` in the stream's dtype, then the final norm and the unembedding,
    one full-vocab row per layer copied to the host, as in vLLM-Lens's jacobian-lens example. The
    maps move to the GPU once per request, on the first hooked layer."""
    slot = {layer: k for k, layer in enumerate(layers)}
    on_device = []

    def read(ctx, h):
        if not _rank0():
            return
        x = h[position].unsqueeze(0)
        if maps is not None:
            if not on_device:
                on_device.append(maps.to(h.device))
            J = on_device[0] if maps.dim() == 2 else on_device[0][slot[ctx.layer_idx]]
            x = x @ J.T
        _append(ctx, "rows", _unembed(ctx, x).cpu())
    return read


@cell("jacobian_lens", family="llama", backend="vllm_lens")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None,
                  check=False):
    """The J-lens readout in the form of vLLM-Lens's jacobian-lens example
    (examples/jacobian_lens.py, `run_jlens`): a Hook on the read layers whose closure carries the
    transport maps to the worker, applies them to the summed stream (hidden + residual after the
    layer), and projects through the final norm and the unembedding there. The maps travel with
    every request (`_prepare_offline_params` pickles each request's hooks), as the example sends
    them; the plugin's persistent hooks (`LLM.register_hooks`) would instead keep a hook on the
    worker for every later request. Output `[n_layers, B, vocab]` in the engine's dtype; a check
    call also checks each request collected one row per read layer."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    hooked = sorted(set(idx))                         # the order the hook fires and appends in
    maps = _transport(model, transport, hooked)
    fns = [_jlens_fn(maps, hooked, locate(model.tokenizer, unit, position)) for unit in units]
    saved = _hooked(model, units, fns, hooked)
    expected = len(hooked) if check else None
    stacked = torch.stack([torch.cat(_rows(s, expected)) for s in saved], dim=1)
    return _in_order(stacked, hooked, idx)                                      # [n_layers, B, vocab]


def _last_row_stack(last):
    """Keep the request's last row at each hooked layer on the GPU and copy the stack to the host
    once, at the deepest hooked layer."""
    def read(ctx, h):
        if not _rank0():
            return
        ctx.saved.setdefault("_rows", []).append(h[-1:])
        if ctx.layer_idx == last:
            ctx.saved["stream"] = torch.cat(ctx.saved.pop("_rows")).cpu()
    return read


@cell("residual_readout", family="llama", backend="vllm_lens")
def residual_readout(be, model, units, *, layers):
    """The residual stream after each listed layer at each prompt's last token,
    `[len(layers), B, d]` in the engine's dtype. A Hook on the listed layers sees the summed
    stream (hidden + residual after the layer, `_worker_ext.py`, `_hook_inner`) of its request's
    token slice and keeps only the last row, so the worker copies `[n_layers, d]` per request to
    the host instead of the whole prompt that `output_residual_stream` copies at every layer."""
    n = model.n_layers
    read = [i % n for i in layers]
    hooked = sorted(set(read))
    saved = _hooked(model, units, [_last_row_stack(hooked[-1])] * len(units), hooked)
    stacked = torch.stack([s["stream"] for s in saved], dim=1)              # [n_hooked, B, d]
    return _in_order(stacked, hooked, read)                                 # [len(layers), B, d]


@cell("attention_pattern", family="llama", backend="vllm_lens")
def attention_pattern(be, model, units, *, layers="all", check=False):
    """vLLM-Lens 1.3.0's attention read (README "Attention pattern capture"): `output_qk` records
    the post-RoPE query and key at the input of each listed layer's attention kernel, each
    request's own token slice (`_worker_ext.py`, `_qk_hook_inner`: a forward pre-hook on vLLM's
    `Attention`), with the kernel's parameters, and `vllm_lens.attention.attention_patterns`
    replays the softmax on the client in fp32 over the request's positions, the prompt's at
    max_tokens=1: `(n_heads, n_prompt, n_prompt)` per layer. The last query's row,
    `[n_layers, B, n_heads, k_len]`: the replay's probabilities for a timed call and, for a check
    call, their log like the reference's, a zero probability floored at the dtype's smallest
    normal first. 1.2.1 has no capture inside a layer."""
    if not _QK:
        raise Unsupported("vLLM-Lens 1.2.1 hooks only decoder layers (`_worker_ext.py`, "
                          "`install_hooks`: register_forward_pre_hook/register_forward_hook on "
                          "each layer of `_get_layers`) and hands a hook the layer's hidden-state "
                          "slice; it has no capture of the attention module's query and key (Q/K "
                          "capture, `output_qk` and `vllm_lens/attention.py`, arrives in 1.3.0)")
    from vllm import SamplingParams
    from vllm_lens.attention import attention_patterns

    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    params = SamplingParams(max_tokens=1, min_tokens=1, **_GREEDY,
                            extra_args={"output_qk": sorted(set(idx))})
    outs = model.llm.generate(list(units), params, use_tqdm=False)
    probs = [[attention_patterns(out.activations, layer=i)[:, -1, :] for out in outs] for i in idx]
    if not check:
        return torch.stack([torch.stack(layer) for layer in probs])
    return torch.stack([torch.stack([torch.log(p.clamp_min(torch.finfo(p.dtype).tiny)) for p in layer])
                        for layer in probs])


def _captured(model, prompts, layers, max_tokens, export):
    """One generate call for `prompts`; with `export`, vLLM-Lens's activation capture at `layers`
    (returned stacked in ascending layer order, `_worker_ext.py`, `_build_payload`)."""
    from vllm import SamplingParams

    extra = {"output_residual_stream": sorted(set(layers))} if export else None
    params = SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens, **_GREEDY, extra_args=extra)
    return model.llm.generate(list(prompts), params, use_tqdm=False)


@cell("activation_harvest", family="llama", backend="vllm_lens")
def activation_harvest(be, model, units, *, layer=None, layers=None, export=True):
    """The documented activation capture, `output_residual_stream`, on one generate call for the
    whole batch. At every forward the worker copies each request's slice of the summed stream
    (hidden + residual after the layer) to host memory (`_worker_ext.py`, `_hook_inner` phase 3),
    and the plugin returns it as `[n_layers, n_prompt, d]` per request at max_tokens=1
    (`_activations_plugin.py`, `_finalize_offline_outputs`). vLLM's `LLM.generate` returns outputs
    in prompt order. `layers` captures every listed layer in the same call and returns
    `[len(layers), tokens, d]` (`layer` is ignored). `export=False` runs the same call without
    capture and returns the greedy tokens."""
    if not export:
        outs = _captured(model, units, [], 1, export)
        return torch.tensor([[out.outputs[0].token_ids[0]] for out in outs])      # [B, 1]
    n = model.n_layers
    read = [i % n for i in (layers if layers is not None else [layer])]
    outs = _captured(model, units, read, 1, export)
    stacked = torch.cat([out.activations["residual_stream"] for out in outs], dim=1)
    stacked = _in_order(stacked, sorted(set(read)), read)            # listed order of the stack
    return stacked if layers is not None else stacked[0]          # [len(layers), tokens, d] or [tokens, d]


@cell("gen_capture", family="llama", backend="vllm_lens")
def gen_capture(be, model, units, *, layer, new_tokens, export=True, check=False):
    """The same capture over a greedy decode: it appends every decode step's row and is trimmed to
    n_prompt + n_gen - 1 positions (`_finalize_offline_outputs`). The rows from the last prompt
    token on are the newest position of each step, the rows the nnsight cell reads; a check call
    also checks each request has `new_tokens` of them."""
    outs = _captured(model, units, [layer], new_tokens, export)
    rows = []
    for out in outs:
        steps = (out.activations["residual_stream"][0][len(out.prompt_token_ids) - 1:] if export
                 else torch.tensor(out.outputs[0].token_ids)[:, None])
        if check and len(steps) != new_tokens:     # an early EOS shortens the decode
            raise RuntimeError(f"collected {len(steps)} per-step rows, expected {new_tokens}")
        rows.append(steps)
    return torch.stack(rows, dim=1)                              # [new_tokens, B, d] or [.., B, 1]

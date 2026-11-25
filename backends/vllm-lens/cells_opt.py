"""vLLM-Lens cells in the fastest documented realization (backend `vllm_lens_opt`).

Identical to the system's cells (backends/vllm-lens/cells.py) except where vLLM-Lens (1.2.1 and
1.3.0) offers a faster or a documented-but-unused path: the logit lens stacks every layer's last
row and reads them out with one unembedding pass at the last layer (instead of one full-vocab pass
per layer), and ablation is expressed through hooks (a pre-hook's input on vLLM's Qwen2 layers is
the previous layer's MLP output, so zeroing it removes that MLP; the attention output is removed
by recomputing the layer's MLP from the stream without it). A call's units go out as one
`generate` over all their prompts, each request hooked on its own token slice with its own
`ctx.saved`, and come back in prompt order; outputs carry the unit axis at dim -2. Readouts are
computed and kept on tensor-parallel rank 0 only (`_rank0`), and timed and check calls split
(`check`), as in cells.py. Every cell but the logit lens, the J-lens and ablation is cells.py's.

Registered for family "llama" (vLLM module paths `model.norm`, `lm_head`; vLLM-Lens 1.2.1 finds
decoder layers only under `model.layers`-style trees, `_get_layers`, and 1.3.0 from the attention
modules vLLM registers, `_discover_layer_modules`).
"""
import sys

import cloudpickle
import torch
import torch.nn.functional as F

from isb.methodologies.jacobian_lens import load_fitted, orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.data import locate

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


def _rows(saved, expected):
    rows = saved.get("rows", [])
    if len(rows) != expected:     # a chunked prefill or an early EOS would change the count
        raise RuntimeError(f"collected {len(rows)} readout rows, expected {expected}")
    return rows


def _logits(saved_dicts):
    """Each request's one readout row -> `[B, vocab]`."""
    return torch.cat([_rows(saved, 1)[0] for saved in saved_dicts])


def _read_last(last):
    def read(ctx, h):
        if ctx.layer_idx == last and _rank0():
            _append(ctx, "rows", _readout(ctx, h))
    return read


@cell("logit_lens", family="llama", backend="vllm_lens_opt")
def logit_lens(be, model, units, *, layers="all", check=False):
    """The logit lens with one unembedding pass per request: each layer's top-1 token id
    `[n_layers, B, 1]` for a timed call, the logits `[n_layers, B, vocab]` in fp32 for a check call."""
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
            logits = F.linear(normed, ctx.get_parameter("lm_head.weight"))
            ctx.saved["lens"] = (logits.float() if check else logits.argmax(-1, keepdim=True)).cpu()

    saved = _hooked(model, units, [lens] * len(units), idx)
    return torch.stack([s["lens"] for s in saved], dim=1)


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


@cell("jacobian_lens", family="llama", backend="vllm_lens_opt")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None,
                  check=False):
    """The J-lens readout stacked like the opt logit lens: a Hook keeps the request's row at
    `position` (an index into its own prompt tokens) from the summed stream of every read layer
    on the GPU, and at the deepest one applies the transport (one matmul for a shared map, one
    batched matmul for per-layer maps), the final norm and the unembedding in one pass, then
    copies `[n_layers, vocab]` to the host. The maps travel with every request in the hook
    closure, as in vLLM-Lens's jacobian-lens example. Output `[n_layers, B, vocab]` in the
    engine's dtype; a check call also checks no prefill was chunked."""
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
                if check and "lens" in ctx.saved:     # a chunked prefill reached the last layer twice
                    ctx.saved["chunked"] = True
                ctx.saved["lens"] = _unembed(ctx, x).cpu()
        return lens

    fns = [lens_at(locate(model.tokenizer, unit, position)) for unit in units]
    saved = _hooked(model, units, fns, hooked)
    if check and any(s.get("chunked") for s in saved):
        raise RuntimeError("a prefill was chunked; the readout position is not in one forward")
    stacked = torch.stack([s["lens"] for s in saved], dim=1)                    # [n_hooked, B, vocab]
    return _in_order(stacked, hooked, idx)                                      # [n_layers, B, vocab]


def _hooks_request(model, prompts, hooks, *, read_index):
    """One `generate` over `prompts` with the same several hooks on every request (each request
    keeps its own contexts); returns each request's saved dict of hook `read_index`."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, **_GREEDY, extra_args={"apply_hooks": hooks})
    outs = model.llm.generate(list(prompts), params, use_tqdm=False)
    return [out.hook_results[str(read_index)] for out in outs]


@cell("ablation", family="llama", backend="vllm_lens_opt")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """Zero `target` at `layer` through hooks (`target="none"`: none). A timed call sends the
    ablation hooks alone and returns each request's generated token, `[B, 1]`; a check call adds
    the last layer's readout and returns the next-token logits, `[B, vocab]`."""
    from vllm_lens import Hook

    last = model.n_layers - 1
    hooks = []
    if target == "mlp":
        if layer + 1 > last:
            raise Unsupported("zeroing the last layer's MLP needs a pre-hook on the layer after it")

        def zero_mlp(ctx, h):
            return torch.zeros_like(h)          # layer+1's hidden input is layer's MLP output

        hooks = [Hook(fn=zero_mlp, layer_indices=[layer + 1], pre=True)]
    elif target == "attn":
        if layer == 0:
            raise Unsupported("the stream before layer 0 is not a decoder-layer output")

        def drop_attn(ctx, h):
            if ctx.layer_idx == layer - 1:
                ctx.saved["_before"] = h                 # the stream entering `layer`
                return None
            before = ctx.saved.pop("_before")
            block = ctx.model.model.layers[layer]
            return before + block.mlp(block.post_attention_layernorm(before))

        hooks = [Hook(fn=drop_attn, layer_indices=[layer - 1, layer])]
    elif target != "none":
        raise ValueError(f"unknown ablation target {target!r}")
    if not check:
        return _generated_tokens(model, units, {"apply_hooks": hooks} if hooks else None)  # [B, 1]
    readout = Hook(fn=_read_last(last), layer_indices=[last])
    return _logits(_hooks_request(model, units, [*hooks, readout], read_index=len(hooks)))


from .cells import (  # noqa: E402
    _generated_tokens, _in_order, activation_harvest, activation_patching, attention_pattern,
    gen_capture, gen_patching, gen_steering, residual_readout, steering)

# Steering is cells.py's: SteeringVectors where they express the strength, a Hook for the mean
# norm (`_steer_args`).
cell("steering", family="llama", backend="vllm_lens_opt")(steering)
# vLLM-Lens has no faster documented form of a steer under decode or of a transplant.
cell("gen_steering", family="llama", backend="vllm_lens_opt")(gen_steering)
cell("activation_patching", family="llama", backend="vllm_lens_opt")(activation_patching)
cell("gen_patching", family="llama", backend="vllm_lens_opt")(gen_patching)
cell("attention_pattern", family="llama", backend="vllm_lens_opt")(attention_pattern)
cell("residual_readout", family="llama", backend="vllm_lens_opt")(residual_readout)
cell("activation_harvest", family="llama", backend="vllm_lens_opt")(activation_harvest)
cell("gen_capture", family="llama", backend="vllm_lens_opt")(gen_capture)

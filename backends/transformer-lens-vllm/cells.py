"""TransformerLens 4.0 vLLM-bridge cells for the cross-system workloads.

Denotation, from transformer-lens 4.0.0 source (`model_bridge/sources/vllm/overlays/decoder_only.py`,
`plugin.py`): on vLLM's fused-residual layers `blocks.{i}.hook_out` reads `hidden + residual` and a
write returns `(modified - residual, residual)`; `blocks.{i}.attn.hook_out` / `mlp.hook_out` are the
`self_attn` / `mlp` module outputs. Interventions are declarative (`suppress`, `scale`, `add`, `set`)
with one width-shaped value per hook. The compiled driver takes one prompt per forward
(`driver.py`, `_normalize_input_ids`: batch_size=1), so a call's units run as one forward each, in
order. A backend booted with `enable_batching` runs the eager batched path instead
(`driver.py`, `_forward_batched`): every unit of a call goes to the bridge as one ragged list of
token ids in one forward, each intervention applies to every prompt of that forward alike, and
captures and logits come back right-padded to the longest prompt, so each row is cut back to its
prompt's length. That path takes no position-scoped intervention (`pos`; `driver.py`,
`_validate_interventions`). Outputs stack the units along dim -2. Returned logits are rebuilt on the
host from the captured final norm and the unembedding weight.

The bridge exposes no unembedding accessor, so the per-layer logit lens and the steering direction
read weights through the driver's `get_param` (the RPC the driver uses for its own logit
reconstruction), reached through the bridge's `_driver` attribute.
"""
import functools
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from isb.methodologies.jacobian_lens import load_fitted, orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import CALIBRATED, resolve_token, calibrated_strengths
from isb.data import locate


def _ids(model, prompt):
    return model.tokenizer(prompt)["input_ids"]


def _wire(be, intervene):
    """`intervene` in the form the driver's RPC delivers intact. A sharded boot runs the engine
    core in its own process (source.py, boot_vllm), so the driver's collective_rpc crosses vLLM's
    msgpack utility channel, which delivers a tensor as [dtype, shape, data] that validate_spec
    rejects; the documented list form arrives intact. One GPU keeps the worker in this process and
    takes the tensor as is."""
    if intervene and be.engine_options.get("tensor_parallel_size", 1) > 1:
        return {name: {k: v.tolist() if isinstance(v, torch.Tensor) else v
                       for k, v in spec.items()} for name, spec in intervene.items()}
    return intervene


def _forward(be, model, prompts, names=(), intervene=None, logits=True):
    """The forwards of `prompts`: each prompt's (logits `[n_tokens, vocab]` or None, its captures
    `{name: [n_tokens, width]}`). One `run_with_cache` call per prompt on the compiled path, one
    for all of them on the batched path. `logits=False` skips the host-side logit rebuild (the
    driver's `return_logits` option)."""
    kwargs = {"names_filter": list(names)}
    intervene = _wire(be, intervene)
    if intervene:
        kwargs["intervene"] = intervene
    if not logits:
        kwargs.update(return_logits=False, return_type=None)
    rows = [_ids(model, prompt) for prompt in prompts]
    if be.batching:
        out, cache = model.bridge.run_with_cache(rows, **kwargs)        # right-padded [B, S_max, .]
        return [(None if out is None else out[b, :len(row)],
                 {name: cache[name][b, :len(row)] for name in names}) for b, row in enumerate(rows)]
    results = []
    for row in rows:
        out, cache = model.bridge.run_with_cache(torch.tensor([row]), **kwargs)
        results.append((None if out is None else out[0], {name: cache[name][0] for name in names}))
    return results


def _final_logits(be, model, prompts, intervene=None):
    """Each prompt's last-position logits: [B, vocab]."""
    results = _forward(be, model, prompts, intervene=intervene)
    return torch.stack([out[-1] for out, _ in results]).float()


def _next_token(be, model, prompts, intervene=None):
    """Each prompt's greedy next token, `[B, 1]` int64, read the documented way: the bridge's
    forward with its logits, then the last position's argmax (docs/source/content/drivers.md;
    demos/vLLM_Bridge_Integration_Test.ipynb, step 6, `logits[0, -1].argmax()`)."""
    return torch.stack([out[-1].argmax(-1, keepdim=True)
                        for out, _ in _forward(be, model, prompts, intervene=intervene)])


def _last_rows(be, model, prompts, names, positions=None, intervene=None):
    """Each prompt's stream at each hook in `names`, at the last position or at `positions` (one
    per prompt): [len(names), B, width], as captured."""
    positions = positions or [-1] * len(prompts)
    results = _forward(be, model, prompts, names, intervene, logits=False)
    return torch.stack([torch.stack([cache[name][pos] for name in names])
                        for (_, cache), pos in zip(results, positions)], dim=1)


def _weight(model, name):
    """A named weight via the driver's get_param, fetched once per loaded model. The unembedding
    falls back to the tied embedding, in the driver's own order."""
    if name not in model.weights:
        driver = model.bridge._driver
        if name == "unembedding":
            weight = driver.get_param("lm_head.weight")
            weight = weight if weight is not None else driver.get_param("model.embed_tokens.weight")
        else:
            weight = driver.get_param(name)
        model.weights[name] = weight.float()
    return model.weights[name]


def _direction(model, target):
    """The unit unembedding row of `target`'s token (`resolve_token`), fp32 on the host, built once
    per loaded model and kept with the fetched weights."""
    key = ("direction", target)
    if key not in model.weights:
        token_id = resolve_token(model.tokenizer, target)
        model.weights[key] = F.normalize(_weight(model, "unembedding")[token_id], dim=0)
    return model.weights[key]


def _lens_logits(model, rows):
    """The final RMSNorm and the unembedding on the host in fp32, as the driver rebuilds its own
    logits. rows: [..., d_model] fp32."""
    norm_weight = _weight(model, "model.norm.weight")
    eps = model.config.rms_norm_eps
    normed = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + eps) * norm_weight
    return normed @ _weight(model, "unembedding").T


def _jlens_maps(model, transport, d, device="cpu"):
    """The Jacobian-lens transport as fp32 on `device`: None (identity), one shared [d, d] map (an
    int seed, `orthogonal`, or a tensor) or {layer: [d, d]} (a fitted lens, `load_fitted`, or a
    mapping). A seeded or fitted map is built once per loaded model and device and kept with the
    fetched weights, so the QR and the checkpoint load stay out of the timed readout."""
    if transport is None:
        return None
    key = f"jlens:{device}:{transport}" if isinstance(transport, (int, str)) else None
    if key in model.weights:
        return model.weights[key]
    source = (load_fitted(transport) if isinstance(transport, str)
              else orthogonal(d, transport) if isinstance(transport, int) else transport)
    maps = ({layer: m.to(device, torch.float32) for layer, m in source.items()}
            if isinstance(source, Mapping) else source.to(device, torch.float32))
    if key is not None:
        model.weights[key] = maps
    return maps


def _transported(rows, maps, idx):
    """`h @ J_l^T` for each layer's rows, before the final norm (isb/methodologies/jacobian_lens.py,
    `_jlens_proxy`). rows: [n_layers, B, d] fp32; idx: the layer of each block of rows."""
    if maps is None:
        return rows
    if isinstance(maps, Mapping):
        return torch.stack([block @ maps[i].T for block, i in zip(rows, idx)])
    return rows @ maps.T


def _steer_layers(model, layer, layers, alpha):
    """The steered layers in forward order; a write at the read-out layer is refused."""
    n, last = model.n_layers, model.n_layers - 1
    sites = sorted({i % n for i in ([layer] if layers is None else layers)})
    if alpha != 0 and last in sites:
        raise ValueError(f"steer layer {last} is the read-out layer; pick layers < {last}")
    return sites


def _steer_edits(be, model, prompts, sites, direction, alpha):
    """Each prompt's `add` edits for a steer at every layer of `sites` (in forward order) with
    strength alpha times the stream's mean per-token norm at that layer. `add` takes a fixed value,
    so the strength is read off a capture forward first, one per steered layer: the stream at a
    layer is read with the edits upstream of it applied, as a live write in one forward would see
    it. The first capture has no edit and runs every prompt together (one forward on the batched
    path); after it each prompt carries its own values, and a batched forward applies one value to
    every prompt, so each prompt then runs alone."""
    edits = [{} for _ in prompts]
    for site in sites:
        name = f"blocks.{site}.hook_out"
        if not edits[0]:
            caches = [cache for _, cache in _forward(be, model, prompts, [name], logits=False)]
        else:
            caches = [_forward(be, model, [prompt], [name], edit, logits=False)[0][1]
                      for prompt, edit in zip(prompts, edits)]
        for edit, cache in zip(edits, caches):
            stream_norm = cache[name].float().norm(dim=-1).mean().item()
            edit[name] = {"op": "add", "value": alpha * stream_norm * direction}
    return edits


def mean_norms(be, model, prompts, layers):
    """The unsteered stream's mean per-token norm at each of `layers` (`blocks.{i}.hook_out`),
    averaged over `prompts`: one capture forward with no intervention."""
    names = [f"blocks.{i}.hook_out" for i in layers]
    caches = [cache for _, cache in _forward(be, model, prompts, names, logits=False)]
    return [sum(c[name].float().norm(dim=-1).mean().item() for c in caches) / len(caches)
            for name in names]


def _fixed_edit(be, model, units, sites, direction, alpha):
    """The one `add` edit every prompt carries under `scale_by="calibrated_mean_norm"`: at each site
    the fixed value `alpha / len(sites) * n * d`, `n` the first site's unsteered mean norm measured
    once per loaded engine (isb/methodologies/steering.py, `calibrated_strengths`). `add` takes exactly such a value, so
    the steered forward needs no capture first and the batched path runs a call as one forward."""
    strengths = calibrated_strengths(be, sites, alpha, units,
                                     functools.partial(mean_norms, be, model))
    return {f"blocks.{site}.hook_out": {"op": "add", "value": k * direction}
            for site, k in zip(sites, strengths)}


def _steer_requests(be, model, units, layer, layers, target, alpha, scale_by):
    """The steered requests of one steering call in unit order, as (prompts, intervene) pairs:
    every unit with no spec at `alpha=0`, every unit with the one fixed edit under
    `scale_by="calibrated_mean_norm"` (`_fixed_edit`), else each unit alone with the edits its
    capture forwards measured (`_steer_edits`). Timed calls and check calls make these requests."""
    sites = _steer_layers(model, layer, layers, alpha)
    if alpha == 0:
        return [(units, None)]
    if scale_by == "token_norm":
        raise Unsupported("'add' carries one width-shaped value per hook; a strength proportional "
                          "to each token's own norm needs a different value at every position "
                          "(intervention_specs.validate_spec)")
    direction = _direction(model, target)
    if scale_by == CALIBRATED:
        return [(units, _fixed_edit(be, model, units, sites, direction, alpha))]
    edits = _steer_edits(be, model, units, sites, direction, alpha / len(sites))
    return [([prompt], edit) for prompt, edit in zip(units, edits)]


def _patch_requests(be, model, units, layer, patch, positions):
    """The patched requests of one patching call in unit order, as (prompts, intervene) pairs: the
    corrupt prompts with no spec at `patch=False`, else each corrupt prompt alone with a `set` of
    its clean prompt's `blocks.{layer}.hook_out` last row at the corrupt run's last position
    (`pos`, which needs boot_vllm(enable_position_interventions=True)), read by a capture forward
    of the clean prompt. Timed calls and check calls make these requests."""
    if not patch:
        return [([unit[1] for unit in units], None)]
    if positions != "last":
        raise Unsupported("an intervention carries one width-shaped value per hook; writing a "
                          "different clean activation at every position in one forward is not "
                          "expressible; a single-position patch is ('set' with 'pos')")
    if be.batching:
        raise Unsupported("the eager batched path applies interventions to the whole tensor and "
                          "takes no 'pos' (driver.py, _validate_interventions; boot_vllm refuses "
                          "enable_position_interventions with enable_batching, source.py)")
    name = f"blocks.{layer}.hook_out"
    requests = []
    for clean_prompt, corrupt_prompt in units:
        (_, cache), = _forward(be, model, [clean_prompt], [name], logits=False)
        last = len(_ids(model, corrupt_prompt)) - 1
        requests.append(([corrupt_prompt],
                         {name: {"op": "set", "value": cache[name][-1], "pos": last}}))
    return requests


@cell("logit_lens", family="llama", backend="transformer_lens")
def logit_lens(be, model, units, *, layers="all"):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the host, as the driver does for its own logits."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _last_rows(be, model, units, [f"blocks.{i}.hook_out" for i in idx]).float()
    return _lens_logits(model, rows)                                         # [n_layers, B, vocab]


@cell("jacobian_lens", family="llama", backend="transformer_lens")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """The logit lens at each prompt's readout position (`locate`) with each layer's transport
    applied to the captured block output before the final RMSNorm and the unembedding, all on the
    host in fp32 as the driver rebuilds its own logits."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else list(layers)
    positions = [locate(model.tokenizer, unit, position) for unit in units]
    names = [f"blocks.{i % n}.hook_out" for i in idx]
    rows = _last_rows(be, model, units, names, positions).float()
    rows = _transported(rows, _jlens_maps(model, transport, rows.shape[-1]), idx)
    return _lens_logits(model, rows)                                         # [n_layers, B, vocab]


@cell("residual_readout", family="llama", backend="transformer_lens")
def residual_readout(be, model, units, *, layers):
    """`blocks.{i}.hook_out` (`hidden + residual` after layer i) at each prompt's last token, in the
    engine's dtype as the driver returns it: [len(layers), B, d_model]."""
    return _last_rows(be, model, units, [f"blocks.{i % model.n_layers}.hook_out" for i in layers])


@cell("steering", family="llama", backend="transformer_lens")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm", check=False):
    """`add` takes a fixed value, so a strength relative to the stream's own norm needs a capture
    forward per steered layer before the steered forward (`_steer_edits`); `scale_by=
    "calibrated_mean_norm"` adds fixed values measured once per loaded engine instead (`_fixed_edit`).
    `layers` steers every listed layer in one forward, `alpha / len(layers)` each. A timed call
    returns each unit's greedy next token `[B, 1]`, a check call the next-token logits `[B, vocab]`,
    from the same requests (at `alpha=0`, the unsteered request)."""
    requests = _steer_requests(be, model, units, layer, layers, target, alpha, scale_by)
    read = _final_logits if check else _next_token
    return torch.cat([read(be, model, prompts, edit) for prompts, edit in requests])


@cell("gen_steering", family="llama", backend="transformer_lens")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); per-step interventions during decoding are not "
                      "expressible")


@cell("activation_patching", family="llama", backend="transformer_lens")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all", check=False):
    """`positions="last"`: capture the clean block output, then `set` it at the corrupt run's last
    position (`_patch_requests`). A timed call returns each unit's greedy next token `[B, 1]`, a
    check call the next-token logits `[B, vocab]`, from the same requests (at `patch=False`, the
    unpatched corrupt request)."""
    requests = _patch_requests(be, model, units, layer, patch, positions)
    read = _final_logits if check else _next_token
    return torch.cat([read(be, model, prompts, edit) for prompts, edit in requests])


@cell("ablation", family="llama", backend="transformer_lens")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """`suppress` zeroes `blocks.{layer}.{target}.hook_out` (the `self_attn` or `mlp` module
    output) at every position; `target="none"` sends no spec. A timed call returns each unit's
    greedy next token `[B, 1]`, a check call the next-token logits `[B, vocab]`, from the same
    requests."""
    site = f"blocks.{layer}.{target}.hook_out"
    intervene = None if target == "none" else {site: {"op": "suppress"}}
    read = _final_logits if check else _next_token
    return read(be, model, units, intervene)


@cell("activation_harvest", family="llama", backend="transformer_lens")
def activation_harvest(be, model, units, *, layer=None, layers=None, export=True):
    """The worker copies each `blocks.{i}.hook_out` buffer to host memory as wire bytes
    (`worker_extension.py`, `tl_read_captures`; the batched path copies every request's rows in the
    hook). Each prompt's valid rows, in prompt order and the engine's dtype: [tokens, d_model], or
    [len(layers), tokens, d_model] for `layers`, which replaces `layer`. `export=False` is the
    baseline: the same requests with no capture and no logit rebuild, so the rows' overhead over it
    is the export's cost. It returns no output: the bridge's forward drops the token vLLM
    generated, and the documented token read (`_next_token`) rebuilds every position's logits on
    the host, which the harvest rows never do."""
    if not export:
        _forward(be, model, units, [], logits=False)
        return None
    names = [f"blocks.{i}.hook_out" for i in ([layer] if layers is None else layers)]
    caches = [cache for _, cache in _forward(be, model, units, names, logits=False)]
    reads = [torch.cat([cache[name] for cache in caches]) for name in names]
    return reads[0] if layers is None else torch.stack(reads)


@cell("attention_pattern", family="llama", backend="transformer_lens")
def attention_pattern(be, model, units, *, layers="all"):
    raise Unsupported("the vLLM overlay lists blocks.{i}.attn.hook_pattern, hook_attn_scores, "
                      "hook_q/hook_k and hook_rot_q/hook_rot_k as non-firing (paged attention "
                      "fuses QK^T and softmax, QKVParallelLinear fuses the projections), and "
                      "attn.hook_in does not fire either, so neither the weights nor the q/k to "
                      "rebuild them are captured (overlays/decoder_only.py, nonfiring_hooks)")


@cell("gen_capture", family="llama", backend="transformer_lens")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); a read at every decode step is not expressible")

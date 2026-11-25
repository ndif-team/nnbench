"""interp-engine cells for the cross-system workloads.

Denotation, the same in interp-engine 1.12.0 and 1.14.3 (`interp_engine/vllm_capture/_hooks.py`,
`_sum_residual`; `vllm_capture/requests.py`, `_mk_resid_post_hook`; `vllm_capture/steering.py`): on
vLLM's fused-residual layers `resid_post.L` reads `hidden + residual` after layer L, and a
`resid_post` steer adds its delta onto the first element, so the next layer's fused add carries it.
`decode_residuals` runs vLLM's own final norm and unembedding in the worker; `unembed_rows` returns
rows of the unembedding matrix.

A call's units go out as independent requests issued together on the facade's loop
(`asyncio.gather`; `capture` is "safe to call concurrently", vllm_backend.py), the form the batch
capture itself takes (capture.py: `_run_with_cache_via_protocol` at 1.12.0, `_capture_via_protocol`
at 1.14.3). `capture_generation` and `generate_steered`, documented as "Single request-locked use",
go out the same way. Every write op acts along given `d_model` directions at every position: `AddSpec`,
`OrthogonalDecompSpec` and `ProjectionCapSpec`, and the lens's norm-scaled add, ablate and swap,
which 1.12.0 takes as `lens_intervention` dicts and 1.13 on as `NormScaledAddSpec`, `AblateSpec`
and `SwapSpec` (steer_specs.py). Workloads that need a per-position value or a zeroed submodule are
declared unsupported. Outputs carry the unit axis at dim -2. A timed call returns the request's own
result: a token is the one the engine generated (`GenStep.token_id`; the completion's `token_ids`
on 1.12.0's lens path), a read the tensor the system hands back; a check call returns what scoring
compares.

From 1.13 on a steer shared by the requests of a call is a `steer()` block (steer.py: "the usual
form"; a served call inside it carries the spec as its own per-request steer), `lens_intervention`
is deprecated in its favour (vllm_backend.py, `_legacy_lens_steering`), and `capture` takes `rows`,
the prompt rows to return (a static worker then sends only those). `_BLOCKS` selects those forms.
"""
import asyncio
import contextlib
from collections.abc import Mapping
from importlib import metadata

import torch
import torch.nn.functional as F

from isb.methodologies.jacobian_lens import load_fitted, orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import resolve_token
from isb.data import locate


# Whether the installed release has `steer()` blocks, `NormScaledAddSpec` and `capture(rows=)`.
_BLOCKS = tuple(map(int, metadata.version("interp-engine").split(".")[:2])) >= (1, 13)


def _ids(model, prompts):
    return [model.tokenizer(prompt)["input_ids"] for prompt in prompts]


def _gather(model, coros):
    """Independent requests issued together on the facade's loop; their results in order."""
    async def all_of():
        return await asyncio.gather(*coros)

    return model.engine.runner.run(all_of())


def _capture(model, ids, layers, specs=None):
    """`resid_post` rows of every prompt for each layer, in the order given: one
    `[n_layers, n_tokens, d_model]` per prompt. `specs` gives each request its own SteeringSpec."""
    from interp_engine import to_address

    points = [f"resid_post.{i}" for i in layers]
    specs = specs or [None] * len(ids)
    engine = model.engine.model
    acts = _gather(model, [engine.capture(row, points, steering_spec=spec)
                           for row, spec in zip(ids, specs)])
    return [torch.stack([a[to_address(p)] for p in points]) for a in acts]


def _rows(model, ids, layers, positions=None, specs=None):
    """`resid_post` of each layer at one row of every prompt (`positions`, default the last):
    `[n_layers, B, d_model]`, one copy of the picked rows. From 1.13 the request names the row
    (`capture(rows=)`), so a static worker sends that row alone; 1.12.0 returns every prompt row
    and the row is picked here."""
    from interp_engine import to_address

    # `rows` counts from the prompt's start (protocol.py, `checked_rows` refuses anything else)
    positions = [len(row) - 1 for row in ids] if positions is None else \
        [pos + len(row) if pos < 0 else pos for pos, row in zip(positions, ids)]
    points = [f"resid_post.{i}" for i in layers]
    specs = specs or [None] * len(ids)
    engine = model.engine.model
    if _BLOCKS:
        acts = _gather(model, [engine.capture(row, points, steering_spec=spec, rows=[pos])
                               for row, spec, pos in zip(ids, specs, positions)])
        picks = [0] * len(ids)                                  # the one row each request returns
    else:
        acts = _gather(model, [engine.capture(row, points, steering_spec=spec)
                               for row, spec in zip(ids, specs)])
        picks = positions
    addresses = [to_address(p) for p in points]
    return torch.stack([a[address][pick] for address in addresses
                        for a, pick in zip(acts, picks)]).unflatten(0, (len(points), len(ids)))


def _final_logits(model, ids, specs=None):
    """Next-token logits of every prompt's last position: the last layer's residual through the
    model's own final norm and unembedding. [B, vocab]"""
    last = _rows(model, ids, [model.n_layers - 1], specs=specs)[0]
    return model.engine.decode_residuals(last).float()


@cell("logit_lens", family="llama", backend="interp_engine")
def logit_lens(be, model, units, *, layers="all", check=False):
    """Each layer's `resid_post` at every prompt's last token through `decode_residuals` (vLLM's
    final norm and unembedding in the worker), which hands the logits to the client: a timed
    call returns each layer's top-1 token id `[n_layers, B, 1]`, a check call the logits in fp32,
    `[n_layers, B, vocab]`."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _rows(model, _ids(model, units), idx)                               # [n_layers, B, d]
    logits = model.engine.decode_residuals(rows.reshape(-1, rows.shape[-1]))
    return (logits.float() if check else logits.argmax(-1, keepdim=True)).reshape(len(idx), len(units), -1)


def _jlens_maps(model, transport, d):
    """The Jacobian-lens transport as fp32 on the client: None (identity), one shared [d, d] map
    (an int seed, `orthogonal`, or a tensor) or {layer: [d, d]} (a fitted lens, `load_fitted`, or
    a mapping). A seeded or fitted map is built once per loaded model, so the QR and the checkpoint
    load stay out of the timed readout."""
    if transport is None:
        return None
    key = transport if isinstance(transport, (int, str)) else None
    if key in model.maps:
        return model.maps[key]
    source = (load_fitted(transport) if isinstance(transport, str)
              else orthogonal(d, transport) if isinstance(transport, int) else transport)
    maps = ({layer: m.float() for layer, m in source.items()} if isinstance(source, Mapping)
            else source.float())
    if key is not None:
        model.maps[key] = maps
    return maps


@cell("jacobian_lens", family="llama", backend="interp_engine")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """interp-engine's Jacobian lens (`layer_logits`, lens.py) applies a caller's transport on the
    client to the fp32 residual (`transport(residual.float(), layer)`) and decodes it with the
    model's own final norm and unembedding, but on vLLM it reads the residual through native
    extraction (`capture_residuals` -> `capture_resid_post`), which needs `enable_extraction` and
    stops one layer short of the last (vllm_capture/native.py), decodes every position, and its
    fused worker read-out (`set_lens_jacobians` + `lens_capture_readout_stream`) returns top-k
    probabilities, not logits. So this cell composes the same steps: `capture` at `resid_post` of
    each layer, the row at each prompt's readout position (`locate`), `h @ J_l^T` in fp32 on the
    client (isb/methodologies/jacobian_lens.py), then `decode_residuals` (vLLM's final norm and
    `compute_logits` in the worker) once over every row. Returns the `[n_layers, B, vocab]` logits
    in the dtype the worker hands back, for both kinds of call."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else list(layers)
    positions = [locate(model.tokenizer, unit, position) for unit in units]
    rows = _rows(model, _ids(model, units), [i % n for i in idx], positions)      # [n_layers, B, d]
    maps = _jlens_maps(model, transport, rows.shape[-1])
    if isinstance(maps, Mapping):
        rows = torch.stack([block.float() @ maps[i].T for block, i in zip(rows, idx)])
    elif maps is not None:
        rows = rows.float() @ maps.T
    logits = model.engine.decode_residuals(rows.reshape(-1, rows.shape[-1]))
    return logits.reshape(len(idx), len(units), -1)                          # [n_layers, B, vocab]


@cell("residual_readout", family="llama", backend="interp_engine")
def residual_readout(be, model, units, *, layers):
    """`resid_post.{i}` (`hidden + residual` after layer i) at each prompt's last token, in the
    engine's dtype as `capture` returns it on CPU: [len(layers), B, d_model]. On 1.12.0 every
    prompt row crosses (its capture takes no position); from 1.13 the request names the row
    (`_rows`)."""
    return _rows(model, _ids(model, units), [i % model.n_layers for i in layers])


@cell("attention_pattern", family="llama", backend="interp_engine")
def attention_pattern(be, model, units, *, layers="all", check=False):
    """`capture_attention` (protocol.py): vLLM's paged kernel never forms the probabilities, so the
    worker records post-RoPE q/k at each layer's attention op (a forward pre-hook on
    `self_attn.attn`, vllm_capture/attn.py) and the client rebuilds the scores, expanding the KV
    heads to the query heads (`repeat_interleave`, the grouping of HF's `repeat_kv`), with the
    causal mask, and their softmax (`recompute_attn_scores`, `attn_probs_from_scores`); one request
    per prompt, issued together (the q/k store is per request, vllm_capture/requests.py). A timed
    call returns the last query's row of `probs` as the client computes it (fp32),
    `[n_layers, B, n_heads, k_len]`; a check call returns its log, like the reference, a zero
    probability floored at the dtype's smallest normal before the log."""
    if be.backend == "vllm-static":
        raise Unsupported("vllm-static serves capture_attention only from Address('attn', layer) "
                          "taps declared at load; static_points='auto' declares resid_post alone, "
                          "and with graphs replayed the hooked path refuses (vllm_backend.py, "
                          "_use_static_attn, _require_hooks)")
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    engine = model.engine.model
    results = _gather(model, [engine.capture_attention(row, idx) for row in _ids(model, units)])
    probs = torch.stack([r[i]["probs"][:, -1, :] for i in idx
                         for r in results]).unflatten(0, (len(idx), len(results)))
    return torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny)) if check else probs


def _direction(model, token_id):
    """The target's unit unembedding row, fp32, read once per loaded engine and token id and kept
    on the model record. On 1.12.0 `unembed_rows` is vLLM-backend surface outside the sync
    protocol (the facade gained it in 1.13), so it runs on the facade's loop through the facade's
    documented `model` property, the same call on every release."""
    if token_id not in model.directions:
        rows = model.engine.runner.run(model.engine.model.unembed_rows([token_id]))
        model.directions[token_id] = F.normalize(rows[0].float(), dim=0)
    return model.directions[token_id]


async def _token_ids(steps):
    """The ids of a `generate_steps` stream's GenSteps, in order."""
    return [step.token_id async for step in steps]


def _token_table(seqs, max_tokens):
    """Each prompt's generated ids, `[B, max_tokens]` int64. A request that stopped at EOS early has
    no token for the later steps, so it raises, as the check calls' per-step reads do."""
    for seq in seqs:
        if len(seq) != max_tokens:
            raise RuntimeError(f"generated {len(seq)} tokens, expected {max_tokens}")
    return torch.tensor(seqs)


def _steps_first(tokens):
    """`[B, steps]` generated ids in a generation's per-step layout, `[steps, B, 1]`."""
    return tokens.T.unsqueeze(-1)


def _next_tokens(model, ids, specs=None, masks=None, *, max_tokens=1):
    """Each prompt's greedy tokens, `[B, max_tokens]` int64: one `generate_steps` request per
    prompt (temperature 0), issued together, steered by its spec in `specs` (leaving the prompt
    positions its `masks` entry lists unsteered), else by an open `steer()` block (1.13 on), else
    not at all. `generate_steps` registers the steer against its own request and yields a GenStep
    per id of vLLM's request output (interp_engine/vllm_backend.py, `VLLMModel.generate_steps`;
    vLLM-backend surface on 1.12.0, a protocol method on 1.14.3); the documented `with steer(model,
    spec): generate_stream(model, tokens)` runs it on vLLM for one prompt (steer.py,
    `generate_stream`)."""
    engine = model.engine.model
    specs = specs or [None] * len(ids)
    masks = masks or [None] * len(ids)
    return _token_table(_gather(model, [_token_ids(engine.generate_steps(
        row, max_tokens=max_tokens, temperature=0.0, steering_spec=spec, position_mask=mask))
        for row, spec, mask in zip(ids, specs, masks)]), max_tokens)


@contextlib.contextmanager
def _lens_steer(model, ids, *, layer, direction, alpha):
    """interp-engine's live norm-relative write on the requests issued inside the block; yields
    each prompt's extra request arguments. The lens's steer adds (strength * ||h||) * d at each
    token from its own norm, at every forward, prompt and generated positions alike; max_fraction
    caps the added norm at max_fraction * ||h||, so it is set above alpha. From 1.13 it is a
    `NormScaledAddSpec` in a `steer()` block around the requests (steer.py; the block's default
    writes generated positions and masks none; vllm_capture/steering.py,
    `_modify_norm_scaled_add`), so a request carries nothing; on 1.12.0 it is each request's
    `lens_intervention` 'steer' op with steer_generated (vllm_capture/lens/intervene.py), the form
    1.13 deprecates."""
    strength, cap = float(alpha), 2.0 * abs(float(alpha))
    if _BLOCKS:
        from interp_engine import NormScaledAddSpec, SteeringSpec, steer

        spec = SteeringSpec.at(f"resid_post.{layer}", NormScaledAddSpec(
            vector=direction, strength=strength, max_fraction=cap))
        with steer(model.engine.model, spec):
            yield [{} for _ in ids]
        return
    specs = [{"op": "steer", "layer": layer, "delta": direction.tolist(), "strength": strength,
              "max_fraction": cap}]
    yield [{"lens_intervention": {"specs": specs, "steer_generated": True, "skip_positions": [],
                                  "prompt_len": len(row)}} for row in ids]


def _lens_steered_tokens(model, ids, *, layer, direction, alpha, max_tokens=1):
    """Each prompt's greedy tokens under the lens's steer (`_lens_steer`), `[B, max_tokens]`
    int64. From 1.13 the requests are `generate_steps` inside the block, as interp-engine's server
    runs a steered generation (interp_engine/server/app.py, `generate`). 1.12.0 takes the lens
    steer per request only on the capture methods (vllm_backend.py, `set_lens_intervention`: "The
    server path uses per-request lens registration inside capture_generation"), so each request is
    a `capture_generation` naming no point (`_assert_points_captured`: "An empty request list
    checks nothing") and the tokens are read off its completion output."""
    with _lens_steer(model, ids, layer=layer, direction=direction, alpha=alpha) as extra:
        if _BLOCKS:
            return _next_tokens(model, ids, max_tokens=max_tokens)
        engine = model.engine.model
        results = _gather(model, [engine.capture_generation(row, [], max_tokens=max_tokens,
                                                            temperature=0.0, **kw)
                                  for row, kw in zip(ids, extra)])
    return _token_table([list(completion.token_ids) for completion, _ in results], max_tokens)


def _lens_steered_logits(model, ids, *, layer, direction, alpha, max_tokens):
    """The next-token logits of each forward of every prompt under the lens's steer
    (`_lens_steer`), with no steer at `alpha=0`: each request captures the last layer's
    `resid_post` through its greedy generation, and the rows from the prompt's last on go through
    `decode_residuals`. [max_tokens, B, vocab]"""
    from interp_engine import to_address

    point = f"resid_post.{model.n_layers - 1}"
    engine = model.engine.model
    steer = (_lens_steer(model, ids, layer=layer, direction=direction, alpha=alpha) if alpha != 0
             else contextlib.nullcontext([{} for _ in ids]))
    with steer as extra:
        results = _gather(model, [engine.capture_generation(row, [point], max_tokens=max_tokens,
                                                            temperature=0.0, **kw)
                                  for row, kw in zip(ids, extra)])
    rows = []
    for row, (_, acts) in zip(ids, results):
        steps = acts[to_address(point)][len(row) - 1:]          # prompt's last row, then each step
        if len(steps) != max_tokens:
            raise RuntimeError(f"collected {len(steps)} readout rows, expected {max_tokens}")
        rows.append(steps)
    stacked = torch.stack(rows, dim=1)                                        # [max_tokens, B, d]
    logits = model.engine.decode_residuals(stacked.reshape(-1, stacked.shape[-1])).float()
    return logits.reshape(max_tokens, len(ids), -1)


def _mean_norm_specs(model, ids, *, layer, direction, alpha):
    """One SteeringSpec per prompt adding alpha * n * d at `layer`, with n the prompt's unsteered
    mean per-token norm there, read by one `capture` request per prompt: AddSpec takes a fixed
    vector, so the norm is measured before the steered request."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

    norms = [a[0].float().norm(dim=-1).mean().item() for a in _capture(model, ids, [layer])]
    return [SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=direction, scale=alpha * norm)])}) for norm in norms]


@cell("steering", family="llama", backend="interp_engine")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm",
             check=False):
    """A timed call returns each unit's greedy next token `[B, 1]` from one steered one-token
    generation per prompt, a check call the next-token logits `[B, vocab]`; at `alpha=0` the
    requests carry no steer. `token_norm` is the lens's norm-scaled steer (`_lens_steer`).
    `mean_norm` has no live form (AddSpec takes a fixed vector), so one request per prompt captures
    the unsteered stream's mean norm at `layer` and a second steers with it (`_mean_norm_specs`)."""
    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    ids = _ids(model, units)
    if alpha == 0:
        return _final_logits(model, ids) if check else _next_tokens(model, ids)
    direction = _direction(model, resolve_token(model.tokenizer, target))
    if scale_by == "token_norm":
        if check:
            return _lens_steered_logits(model, ids, layer=layer, direction=direction, alpha=alpha,
                                        max_tokens=1)[0]
        return _lens_steered_tokens(model, ids, layer=layer, direction=direction, alpha=alpha)
    specs = _mean_norm_specs(model, ids, layer=layer, direction=direction, alpha=alpha)
    return _final_logits(model, ids, specs) if check else _next_tokens(model, ids, specs)


@cell("gen_steering", family="llama", backend="interp_engine")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm", check=False):
    """The lens's norm-scaled steer (`_lens_steer`) over a greedy generation of `new_tokens` per
    prompt: a timed call returns the generated tokens `[new_tokens, B, 1]`, a check call each
    step's next-token logits `[new_tokens, B, vocab]`. At `alpha=0` the requests carry no steer
    whatever the scale. A strength from the mean norm is unsupported."""
    ids = _ids(model, units)
    if alpha == 0:
        if check:
            return _lens_steered_logits(model, ids, layer=layer, direction=None, alpha=0,
                                        max_tokens=new_tokens)
        return _steps_first(_next_tokens(model, ids, max_tokens=new_tokens))
    if scale_by != "token_norm":
        raise Unsupported("the live norm-relative write (the lens's steer: NormScaledAddSpec, "
                          "lens_intervention 'steer' on 1.12.0) scales by each token's own norm; a "
                          "strength from the mean norm over the forward's tokens, which differs at "
                          "the prefill step, has no live form (vllm_capture/steering.py)")
    direction = _direction(model, resolve_token(model.tokenizer, target))
    if check:
        return _lens_steered_logits(model, ids, layer=layer, direction=direction, alpha=alpha,
                                    max_tokens=new_tokens)
    return _steps_first(_lens_steered_tokens(model, ids, layer=layer, direction=direction,
                                             alpha=alpha, max_tokens=new_tokens))


@cell("activation_patching", family="llama", backend="interp_engine")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all", check=False):
    """A timed call returns the greedy next token `[B, 1]` of each corrupt prompt's patched
    one-token generation (`generate_steps`), a check call its next-token logits `[B, vocab]`;
    `patch=False` runs the corrupt prompts unpatched. `positions="last"` adds (clean - corrupt) at
    the last position only, an AddSpec whose position_mask excludes every other prompt position;
    the check call runs it through `generate_steered`, capturing the last layer's `resid_post` on
    the same request."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

    clean_ids = _ids(model, [unit[0] for unit in units])
    corrupt_ids = _ids(model, [unit[1] for unit in units])
    if not patch:                                                    # the corrupt runs, unpatched
        return _final_logits(model, corrupt_ids) if check else _next_tokens(model, corrupt_ids)
    if positions != "last":
        raise Unsupported("each write op carries one d_model vector and a request has one position "
                          "mask, so a different value at every position is not expressible; a "
                          "single-position patch is (a masked AddSpec of clean minus corrupt)")
    n = len(units)
    last = _rows(model, clean_ids + corrupt_ids, [layer])[0].float()         # [2B, d_model]
    specs = [SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=last[i] - last[n + i], scale=1.0)])}) for i in range(n)]
    masks = [list(range(len(row) - 1)) for row in corrupt_ids]
    if not check:
        return _next_tokens(model, corrupt_ids, specs, masks)
    from interp_engine import to_address
    from vllm import SamplingParams

    point = f"resid_post.{model.n_layers - 1}"
    captured = [{} for _ in units]
    engine = model.engine.model
    _gather(model, [engine.generate_steered(
        row, SamplingParams(max_tokens=1, temperature=0.0), steering_spec=spec, position_mask=mask,
        capture_points=[point], capture_out=out)
        for row, spec, mask, out in zip(corrupt_ids, specs, masks, captured)])
    rows = torch.stack([out[to_address(point)][-1] for out in captured])
    return model.engine.decode_residuals(rows).float()                       # [B, vocab]


@cell("ablation", family="llama", backend="interp_engine")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """`target="none"` (the baseline) runs the prompts unablated: a timed call returns each unit's
    greedy next token `[B, 1]`, a check call the next-token logits `[B, vocab]`. Zeroing a
    component is unsupported."""
    if target != "none":
        raise Unsupported(f"{target}_out is writable per request, but every write op acts along "
                          f"given directions (add, norm-scaled add, orthogonal rescale, projection "
                          f"cap, ablate, swap); zeroing the whole output would take d_model "
                          f"single-direction removals")
    ids = _ids(model, units)
    return _final_logits(model, ids) if check else _next_tokens(model, ids)


@cell("activation_harvest", family="llama", backend="interp_engine")
def activation_harvest(be, model, units, *, layer=None, layers=None, export=True):
    """The documented batch capture, `run_with_cache` over a padded batch and its mask
    (`interp_engine/capture.py`; from 1.13 the same function is also named `capture`, and
    `run_with_cache` stays as its alias): on vLLM each row runs unpadded as its own `capture`
    request, the requests run concurrently for the scheduler to batch, and each returns on CPU;
    the Cache pads them to [batch, seq, d_model]. Returns the valid rows concatenated in prompt
    order, one copy: [tokens, d_model], or [len(layers), tokens, d_model] for `layers`, which
    replaces `layer` (one request per prompt captures every listed point). `export=False` runs the
    same concurrent requests for one greedy token, since `run_with_cache` needs a capture point,
    and returns the engine's tokens `[B, 1]` (`generate_steps`)."""
    ids = _ids(model, units)
    if not export:
        return _next_tokens(model, ids)
    from interp_engine import run_with_cache

    width = max(map(len, ids))
    tokens = torch.tensor([row + [0] * (width - len(row)) for row in ids])
    mask = (torch.arange(width) < torch.tensor([len(row) for row in ids])[:, None]).long()
    points = [f"resid_post.{i}" for i in ([layer] if layers is None else layers)]
    cache = run_with_cache(model.engine.model, tokens, points, attention_mask=mask)
    valid = torch.cat([cache[point][b, :len(row)] for point in points
                       for b, row in enumerate(ids)])            # [n_points * tokens, d_model]
    return valid if layers is None else valid.unflatten(0, (len(points), -1))


@cell("gen_capture", family="llama", backend="interp_engine")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    """`capture_generation` returns the point at prompt and generated positions,
    [prompt_len + generated_len - 1, d_model] on CPU (`protocol.py`); the rows from the last prompt
    token on are the newest position of each decode step, returned as [new_tokens, B, d_model].
    `export=False` is the same greedy decode with no capture and returns the engine's tokens
    `[new_tokens, B, 1]` (`generate_steps`)."""
    ids = _ids(model, units)
    engine = model.engine.model
    if not export:
        return _steps_first(_next_tokens(model, ids, max_tokens=new_tokens))
    from interp_engine import to_address

    point = f"resid_post.{layer}"
    results = _gather(model, [engine.capture_generation(row, [point], max_tokens=new_tokens,
                                                        temperature=0.0) for row in ids])
    rows = []
    for row, (_, acts) in zip(ids, results):
        steps = acts[to_address(point)][len(row) - 1:]
        if len(steps) != new_tokens:     # an early EOS shortens the decode
            raise RuntimeError(f"collected {len(steps)} per-step rows, expected {new_tokens}")
        rows.append(steps)
    return torch.stack(rows, dim=1)                                           # [new_tokens, B, d_model]

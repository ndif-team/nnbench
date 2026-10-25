"""interp-engine cells for the cross-system workloads.

Denotation, from interp-engine 1.12.0 source (`interp_engine/vllm_capture/requests.py`,
`vllm_capture/steering.py`): on vLLM's fused-residual layers `resid_post.L` reads
`hidden + residual` after layer L, and a `resid_post` steer adds its delta onto the first element,
so the next layer's fused add carries it. `decode_residuals` runs vLLM's own final norm and
unembedding in the worker; `unembed_rows` returns rows of the unembedding matrix.

A call's units go out as independent requests issued together on the facade's loop
(`asyncio.gather`; `capture` is "safe to call concurrently", vllm_backend.py), the form
`run_with_cache` itself takes for a batch (capture.py, `_run_with_cache_via_protocol`); the
request-locked methods (`capture_generation`, `generate_steered`) take turns inside the engine.
Writes are `AddSpec`, `OrthogonalDecompSpec` and `ProjectionCapSpec`, each one `d_model` vector per
layer applied at every position. Workloads that need a per-position value or a zeroed submodule are
declared unsupported. Outputs carry the unit axis at dim -2.
"""
import asyncio

import torch
import torch.nn.functional as F

from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import _resolve_token


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


def _final_logits(model, ids, specs=None):
    """Next-token logits of every prompt's last position: the last layer's residual through the
    model's own final norm and unembedding. [B, vocab]"""
    acts = _capture(model, ids, [model.n_layers - 1], specs)
    return model.engine.decode_residuals(torch.stack([a[0, -1] for a in acts])).float()


@cell("logit_lens", family="llama", backend="interp_engine")
def logit_lens(be, model, units, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    acts = _capture(model, _ids(model, units), idx)
    rows = torch.stack([a[:, -1, :] for a in acts], dim=1)                     # [n_layers, B, d]
    logits = model.engine.decode_residuals(rows.reshape(-1, rows.shape[-1])).float()
    return logits.reshape(len(idx), len(units), -1)                            # [n_layers, B, vocab]


def _direction(model, token_id):
    """The target's unit unembedding row. unembed_rows is vLLM-backend surface outside the sync
    protocol, so it runs on the facade's loop through the facade's documented `model` property."""
    rows = model.engine.runner.run(model.engine.model.unembed_rows([token_id]))
    return F.normalize(rows[0].float(), dim=0)


def _lens_steered_logits(model, ids, *, layer, direction, alpha, max_tokens):
    """interp-engine's live norm-relative write: the per-request lens 'steer' op adds
    (strength * ||h||) * d at each token from its own norm, at every forward with steer_generated
    (vllm_capture/lens/intervene.py). max_fraction caps the added norm at max_fraction * ||h||, so
    it is set above alpha. Returns the next-token logits of each forward of every prompt:
    [max_tokens, B, vocab]."""
    from interp_engine import to_address

    point = f"resid_post.{model.n_layers - 1}"
    engine = model.engine.model

    def lens_of(row):
        return {"specs": [{"op": "steer", "layer": layer, "delta": direction.tolist(),
                           "strength": float(alpha), "max_fraction": 2.0 * abs(float(alpha))}],
                "steer_generated": True, "skip_positions": [], "prompt_len": len(row)}

    results = _gather(model, [engine.capture_generation(row, [point], max_tokens=max_tokens,
                                                        temperature=0.0, lens_intervention=lens_of(row))
                              for row in ids])
    rows = []
    for row, (_, acts) in zip(ids, results):
        steps = acts[to_address(point)][len(row) - 1:]          # prompt's last row, then each step
        if len(steps) != max_tokens:
            raise RuntimeError(f"collected {len(steps)} readout rows, expected {max_tokens}")
        rows.append(steps)
    stacked = torch.stack(rows, dim=1)                                        # [max_tokens, B, d]
    logits = model.engine.decode_residuals(stacked.reshape(-1, stacked.shape[-1])).float()
    return logits.reshape(max_tokens, len(ids), -1)


@cell("steering", family="llama", backend="interp_engine")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    """`token_norm` is the lens 'steer' op. `mean_norm` has no live form: AddSpec takes a fixed
    vector, so one request per prompt captures the unsteered stream at `layer` for its mean
    per-token norm and a second steers with that strength and reads the result."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    ids = _ids(model, units)
    if alpha == 0:
        return _final_logits(model, ids)
    direction = _direction(model, _resolve_token(model.tokenizer, target))
    if scale_by == "token_norm":
        return _lens_steered_logits(model, ids, layer=layer, direction=direction, alpha=alpha,
                                    max_tokens=1)[0]
    norms = [a[0].float().norm(dim=-1).mean().item() for a in _capture(model, ids, [layer])]
    specs = [SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=direction, scale=alpha * norm)])}) for norm in norms]
    return _final_logits(model, ids, specs)


@cell("gen_steering", family="llama", backend="interp_engine")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    if scale_by != "token_norm":
        raise Unsupported("the live norm-relative write (lens 'steer', steer_generated) scales by "
                          "each token's own norm; a strength from the mean norm over the forward's "
                          "tokens, which differs at the prefill step, has no live form "
                          "(vllm_capture/lens/intervene.py)")
    direction = _direction(model, _resolve_token(model.tokenizer, target))
    return _lens_steered_logits(model, _ids(model, units), layer=layer, direction=direction,
                                alpha=alpha, max_tokens=new_tokens)


@cell("activation_patching", family="llama", backend="interp_engine")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all"):
    """`positions="last"`: add (clean - corrupt) at the last position only, a masked AddSpec through
    generate_steered, whose position_mask excludes every other prompt position."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec, to_address
    from vllm import SamplingParams

    clean_ids = _ids(model, [unit[0] for unit in units])
    corrupt_ids = _ids(model, [unit[1] for unit in units])
    if not patch:
        return _final_logits(model, corrupt_ids)                          # the corrupt runs, unpatched
    if positions != "last":
        raise Unsupported("each write op carries one d_model vector and a request has one position "
                          "mask, so a different value at every position is not expressible; a "
                          "single-position patch is (a masked AddSpec of clean minus corrupt)")
    n = len(units)
    acts = _capture(model, clean_ids + corrupt_ids, [layer])
    deltas = [acts[i][0, -1].float() - acts[n + i][0, -1].float() for i in range(n)]
    point = f"resid_post.{model.n_layers - 1}"
    captured = [{} for _ in units]
    engine = model.engine.model
    coros = []
    for row, delta, out in zip(corrupt_ids, deltas, captured):
        spec = SteeringSpec(layers={layer: LayerSteeringSpec(
            operations=[AddSpec(vector=delta, scale=1.0)])})
        coros.append(engine.generate_steered(
            row, SamplingParams(max_tokens=1, temperature=0.0), steering_spec=spec,
            position_mask=list(range(len(row) - 1)), capture_points=[point], capture_out=out))
    _gather(model, coros)
    rows = torch.stack([out[to_address(point)][-1] for out in captured])
    return model.engine.decode_residuals(rows).float()                       # [B, vocab]


@cell("ablation", family="llama", backend="interp_engine")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target != "none":
        raise Unsupported(f"{target}_out is writable per request, but every write op acts along "
                          f"one direction (add, orthogonal rescale, projection cap, lens ablate); "
                          f"zeroing the whole output would take d_model single-direction removals")
    return _final_logits(model, _ids(model, units))


@cell("activation_harvest", family="llama", backend="interp_engine")
def activation_harvest(be, model, units, *, layer, export=True):
    """The documented batch capture, `run_with_cache` over a padded batch and its mask
    (`interp_engine/capture.py`): on vLLM each row runs unpadded as its own `capture` request, the
    requests run concurrently for the scheduler to batch, and each returns on CPU; the Cache pads
    them to [batch, seq, d_model]. The valid rows are concatenated in prompt order. `export=False`
    sends the same concurrent requests through `generate_text` for one greedy token, since
    `run_with_cache` needs a capture point."""
    from interp_engine import run_with_cache

    ids = _ids(model, units)
    if not export:
        return _gather(model, [model.engine.model.generate_text(row, max_tokens=1, temperature=0.0)
                               for row in ids])
    width = max(map(len, ids))
    tokens = torch.tensor([row + [0] * (width - len(row)) for row in ids])
    mask = (torch.arange(width) < torch.tensor([len(row) for row in ids])[:, None]).long()
    point = f"resid_post.{layer}"
    acts = run_with_cache(model.engine.model, tokens, [point], attention_mask=mask)[point]
    return torch.cat([acts[b, :len(row)] for b, row in enumerate(ids)])       # [tokens, d_model]


@cell("gen_capture", family="llama", backend="interp_engine")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    """`capture_generation` returns the point at prompt and generated positions,
    [prompt_len + generated_len - 1, d_model] on CPU (`protocol.py`); the rows from the last prompt
    token on are the newest position of each decode step. `export=False` is the same greedy decode
    through `generate_text`."""
    from interp_engine import to_address

    ids = _ids(model, units)
    engine = model.engine.model
    if not export:
        return _gather(model, [engine.generate_text(row, max_tokens=new_tokens, temperature=0.0)
                               for row in ids])
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

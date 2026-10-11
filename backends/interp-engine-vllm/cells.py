"""interp-engine cells for the comparison specs (design.md §12.14).

Denotation, from interp-engine 1.12.0 source (`interp_engine/vllm_capture/requests.py`,
`vllm_capture/steering.py`): on vLLM's fused-residual layers `resid_post.L` reads
`hidden + residual` after layer L, and a `resid_post` steer adds its delta onto the first element,
so the next layer's fused add carries it. `decode_residuals` runs vLLM's own final norm and
unembedding in the worker; `unembed_rows` returns rows of the unembedding matrix.

Writes are `AddSpec`, `OrthogonalDecompSpec` and `ProjectionCapSpec`, each one `d_model` vector per
layer applied at every position. Workloads that need a per-position value or a zeroed submodule are
declared unsupported. Outputs match the nnsight cells' shapes.
"""
import torch
import torch.nn.functional as F

from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import _resolve_token


def _ids(model, prompt):
    return model.tokenizer(prompt)["input_ids"]


def _capture(model, ids, layers, steering_spec=None):
    """`resid_post` rows for each layer, in the order given: [n_layers, n_tokens, d_model]."""
    from interp_engine import to_address

    points = [f"resid_post.{i}" for i in layers]
    acts = model.engine.capture(ids, points, steering_spec=steering_spec)
    return torch.stack([acts[to_address(p)] for p in points])


def _capture_pair(model, ids_a, ids_b, layer):
    """Two independent captures as concurrent requests on the engine's own loop
    (`capture` is "safe to call concurrently", vllm_backend.py), one round trip instead of two."""
    import asyncio

    from interp_engine import to_address

    point = f"resid_post.{layer}"

    async def both():
        m = model.engine.model
        return await asyncio.gather(m.capture(ids_a, [point]), m.capture(ids_b, [point]))

    acts_a, acts_b = model.engine.runner.run(both())
    return torch.stack([acts_a[to_address(point)]]), torch.stack([acts_b[to_address(point)]])


def _final_logits(model, ids, steering_spec=None):
    """Next-token logits of the last position: the last layer's residual through the model's own
    final norm and unembedding. [1, vocab]"""
    resid = _capture(model, ids, [model.n_layers - 1], steering_spec)[0]
    return model.engine.decode_residuals(resid[-1:]).float()


@cell("logit_lens", family="llama", backend="interp_engine")
def logit_lens(be, model, prompts, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    last_rows = _capture(model, _ids(model, prompts[0]), idx)[:, -1, :]    # [n_layers, d_model]
    return model.engine.decode_residuals(last_rows).float().unsqueeze(1)   # [n_layers, 1, vocab]


def _direction(model, token_id):
    """The target's unit unembedding row. unembed_rows is vLLM-backend surface outside the sync
    protocol, so it runs on the facade's loop through the facade's documented `model` property."""
    rows = model.engine.runner.run(model.engine.model.unembed_rows([token_id]))
    return F.normalize(rows[0].float(), dim=0)


def _lens_steered_logits(model, ids, *, layer, direction, alpha, max_tokens):
    """interp-engine's live norm-relative write: the per-request lens 'steer' op adds
    (strength * ||h||) * d at each token from its own norm, at every forward with steer_generated
    (vllm_capture/lens/intervene.py). max_fraction caps the added norm at max_fraction * ||h||, so
    it is set above alpha. Returns the next-token logits of each forward: [max_tokens, vocab]."""
    from interp_engine import to_address

    point = f"resid_post.{model.n_layers - 1}"
    lens = {"specs": [{"op": "steer", "layer": layer, "delta": direction.tolist(),
                       "strength": float(alpha), "max_fraction": 2.0 * abs(float(alpha))}],
            "steer_generated": True, "skip_positions": [], "prompt_len": len(ids)}
    _, acts = model.engine.runner.run(model.engine.model.capture_generation(
        ids, [point], max_tokens=max_tokens, temperature=0.0, lens_intervention=lens))
    rows = acts[to_address(point)][len(ids) - 1:]           # prompt's last row, then each step
    if len(rows) != max_tokens:
        raise RuntimeError(f"collected {len(rows)} readout rows, expected {max_tokens}")
    return model.engine.decode_residuals(rows).float()


@cell("steering", family="llama", backend="interp_engine")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    """`token_norm` is the lens 'steer' op. `mean_norm` has no live form: AddSpec takes a fixed
    vector, so one request captures the unsteered stream at `layer` for its mean per-token norm and
    a second steers with that strength and reads the result."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    ids = _ids(model, prompts[0])
    if alpha == 0:
        return _final_logits(model, ids)
    direction = _direction(model, _resolve_token(model.tokenizer, target))
    if scale_by == "token_norm":
        return _lens_steered_logits(model, ids, layer=layer, direction=direction, alpha=alpha,
                                    max_tokens=1)
    stream_norm = _capture(model, ids, [layer])[0].float().norm(dim=-1).mean().item()
    spec = SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=direction, scale=alpha * stream_norm)])})
    return _final_logits(model, ids, spec)


@cell("gen_steering", family="llama", backend="interp_engine")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    if scale_by != "token_norm":
        raise Unsupported("the live norm-relative write (lens 'steer', steer_generated) scales by "
                          "each token's own norm; a strength from the mean norm over the forward's "
                          "tokens, which differs at the prefill step, has no live form "
                          "(vllm_capture/lens/intervene.py)")
    direction = _direction(model, _resolve_token(model.tokenizer, target))
    return _lens_steered_logits(model, _ids(model, prompts[0]), layer=layer, direction=direction,
                                alpha=alpha, max_tokens=new_tokens)


@cell("activation_patching", family="llama", backend="interp_engine")
def activation_patching(be, model, prompts, *, layer=6, patch=True, positions="all"):
    """`positions="last"`: add (clean - corrupt) at the last position only, a masked AddSpec through
    generate_steered, whose position_mask excludes every other prompt position."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec, to_address
    from vllm import SamplingParams

    clean_ids, corrupt_ids = _ids(model, prompts[0]), _ids(model, prompts[1])
    if not patch:
        return _final_logits(model, corrupt_ids)                          # the corrupt run, unpatched
    if positions != "last":
        raise Unsupported("each write op carries one d_model vector and a request has one position "
                          "mask, so a different value at every position is not expressible; a "
                          "single-position patch is (a masked AddSpec of clean minus corrupt)")
    if model.concurrent:
        clean, corrupt = (a[0][-1].float() for a in _capture_pair(model, clean_ids, corrupt_ids, layer))
    else:
        clean = _capture(model, clean_ids, [layer])[0][-1].float()
        corrupt = _capture(model, corrupt_ids, [layer])[0][-1].float()
    spec = SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=clean - corrupt, scale=1.0)])})
    point, captured = f"resid_post.{model.n_layers - 1}", {}
    model.engine.runner.run(model.engine.model.generate_steered(
        corrupt_ids, SamplingParams(max_tokens=1, temperature=0.0), steering_spec=spec,
        position_mask=list(range(len(corrupt_ids) - 1)), capture_points=[point],
        capture_out=captured))
    return model.engine.decode_residuals(captured[to_address(point)][-1:]).float()


@cell("ablation", family="llama", backend="interp_engine")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    if target != "none":
        raise Unsupported(f"{target}_out is writable per request, but every write op acts along "
                          f"one direction (add, orthogonal rescale, projection cap, lens ablate); "
                          f"zeroing the whole output would take d_model single-direction removals")
    return _final_logits(model, _ids(model, prompts[0]))

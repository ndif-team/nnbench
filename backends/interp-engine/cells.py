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


@cell("steering", family="llama", backend="interp_engine")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0):
    """AddSpec takes a fixed vector, so a strength relative to the stream's own norm needs two
    requests: one captures the unsteered stream at `layer` for its mean per-token norm, the second
    steers with that strength and reads the result."""
    from interp_engine import AddSpec, LayerSteeringSpec, SteeringSpec

    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    ids = _ids(model, prompts[0])
    if alpha == 0:
        return _final_logits(model, ids)
    stream_norm = _capture(model, ids, [layer])[0].float().norm(dim=-1).mean().item()
    token_id = _resolve_token(model.tokenizer, target)
    # unembed_rows is vLLM-backend surface outside the sync protocol; run it on the facade's loop
    rows = model.engine.runner.run(model.engine.model.unembed_rows([token_id]))
    direction = F.normalize(rows[0].float(), dim=0)
    spec = SteeringSpec(layers={layer: LayerSteeringSpec(
        operations=[AddSpec(vector=direction, scale=alpha * stream_norm)])})
    return _final_logits(model, ids, spec)


@cell("gen_steering", family="llama", backend="interp_engine")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8):
    raise Unsupported("a steering vector and its strength are fixed when the request is built; a "
                      "strength relative to the live residual norm at each decode step is not "
                      "expressible, and generated-step logits are not returned on vLLM")


@cell("activation_patching", family="llama", backend="interp_engine")
def activation_patching(be, model, prompts, *, layer=6, patch=True):
    if patch:
        raise Unsupported("writes are one d_model vector per layer applied at every position; "
                          "replacing each position's residual with another run's is not expressible")
    return _final_logits(model, _ids(model, prompts[1]))                  # the corrupt run, unpatched


@cell("ablation", family="llama", backend="interp_engine")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    if target != "none":
        raise Unsupported(f"writes add, rescale or cap along one direction; zeroing the {target} "
                          f"output is not expressible")
    return _final_logits(model, _ids(model, prompts[0]))

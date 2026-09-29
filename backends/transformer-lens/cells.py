"""TransformerLens 4.0 vLLM-bridge cells for the comparison specs (design.md §12.14).

Denotation, from transformer-lens 4.0.0 source (`model_bridge/sources/vllm/overlays/decoder_only.py`,
`plugin.py`): on vLLM's fused-residual layers `blocks.{i}.hook_out` reads `hidden + residual` and a
write returns `(modified - residual, residual)`; `blocks.{i}.attn.hook_out` / `mlp.hook_out` are the
`self_attn` / `mlp` module outputs. Interventions are declarative (`suppress`, `scale`, `add`, `set`)
with one width-shaped value per hook, and each call is one forward. Returned logits are rebuilt on
the host from the captured final norm and the unembedding weight.

The bridge exposes no unembedding accessor, so the per-layer logit lens and the steering direction
read weights through the driver's `get_param` (the RPC the driver uses for its own logit
reconstruction), reached through the bridge's `_driver` attribute.
"""
import torch
import torch.nn.functional as F

from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import _resolve_token


def _ids(model, prompt):
    return torch.tensor([model.tokenizer(prompt)["input_ids"]])


def _run(model, prompt, names=(), intervene=None, logits=True):
    """One forward. `logits=False` skips the host-side logit rebuild on capture-only passes
    (the driver's `return_logits` option)."""
    kwargs = {"names_filter": list(names)}
    if intervene:
        kwargs["intervene"] = intervene
    if not logits:
        kwargs.update(return_logits=False, return_type=None)
    return model.bridge.run_with_cache(_ids(model, prompt), **kwargs)


def _final_logits(model, prompt, intervene=None):
    logits, _ = _run(model, prompt, intervene=intervene)
    return logits[0, -1:, :].float()                                     # [1, vocab]


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


@cell("logit_lens", family="llama", backend="transformer_lens")
def logit_lens(be, model, prompts, *, layers="all"):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the host, as the driver does for its own logits."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    names = [f"blocks.{i}.hook_out" for i in idx]
    _, cache = _run(model, prompts[0], names, logits=False)
    rows = torch.stack([cache[name][0, -1, :] for name in names]).float()   # [n_layers, d_model]
    norm_weight = _weight(model, "model.norm.weight")
    eps = model.config.rms_norm_eps
    normed = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + eps) * norm_weight
    return (normed @ _weight(model, "unembedding").T).unsqueeze(1)         # [n_layers, 1, vocab]


@cell("steering", family="llama", backend="transformer_lens")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0):
    """`add` takes a fixed value, so a strength relative to the stream's own norm needs two
    forwards: one captures the unsteered block output for its mean per-token norm, the second adds
    the vector and returns the steered logits."""
    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    if alpha == 0:
        return _final_logits(model, prompts[0])
    name = f"blocks.{layer}.hook_out"
    _, cache = _run(model, prompts[0], [name], logits=False)
    stream_norm = cache[name][0].float().norm(dim=-1).mean().item()
    token_id = _resolve_token(model.tokenizer, target)
    direction = F.normalize(_weight(model, "unembedding")[token_id], dim=0)
    return _final_logits(model, prompts[0],
                         {name: {"op": "add", "value": alpha * stream_norm * direction}})


@cell("gen_steering", family="llama", backend="transformer_lens")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); per-step interventions during decoding are not "
                      "expressible")


@cell("activation_patching", family="llama", backend="transformer_lens")
def activation_patching(be, model, prompts, *, layer=6, patch=True):
    if patch:
        raise Unsupported("an intervention carries one width-shaped value per hook; writing a "
                          "different clean activation at every position in one forward is not "
                          "expressible")
    return _final_logits(model, prompts[1])                               # the corrupt run, unpatched


@cell("ablation", family="llama", backend="transformer_lens")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(model, prompts[0])
    return _final_logits(model, prompts[0], {f"blocks.{layer}.{target}.hook_out": {"op": "suppress"}})

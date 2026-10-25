"""TransformerLens 4.0 vLLM-bridge cells for the cross-system workloads.

Denotation, from transformer-lens 4.0.0 source (`model_bridge/sources/vllm/overlays/decoder_only.py`,
`plugin.py`): on vLLM's fused-residual layers `blocks.{i}.hook_out` reads `hidden + residual` and a
write returns `(modified - residual, residual)`; `blocks.{i}.attn.hook_out` / `mlp.hook_out` are the
`self_attn` / `mlp` module outputs. Interventions are declarative (`suppress`, `scale`, `add`, `set`)
with one width-shaped value per hook, and each call is one forward over one prompt: the compiled
driver takes `batch_size=1` (`driver.py`, `_normalize_input_ids`; its eager batched path needs
`boot_vllm(enable_batching=True)`, which excludes the position-scoped interventions these cells
use), so a call's units run as one forward each, in order, and stack along the unit axis (dim -2).
Returned logits are rebuilt on the host from the captured final norm and the unembedding weight.

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


def _final_logits(model, prompts, intervene=None):
    """Each prompt's last-position logits, one forward each: [B, vocab]."""
    return torch.cat([_run(model, prompt, intervene=intervene)[0][0, -1:, :].float()
                      for prompt in prompts])


def _last_rows(model, prompt, names):
    """One prompt's last-position stream at each hook in `names`: [len(names), d_model]."""
    _, cache = _run(model, prompt, names, logits=False)
    return torch.stack([cache[name][0, -1, :] for name in names])


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
def logit_lens(be, model, units, *, layers="all"):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the host, as the driver does for its own logits."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    names = [f"blocks.{i}.hook_out" for i in idx]
    rows = torch.stack([_last_rows(model, prompt, names) for prompt in units], dim=1).float()
    norm_weight = _weight(model, "model.norm.weight")                        # rows: [n_layers, B, d]
    eps = model.config.rms_norm_eps
    normed = rows * torch.rsqrt(rows.pow(2).mean(-1, keepdim=True) + eps) * norm_weight
    return normed @ _weight(model, "unembedding").T                          # [n_layers, B, vocab]


@cell("steering", family="llama", backend="transformer_lens")
def steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    """`add` takes a fixed value, so a strength relative to the stream's own norm needs two
    forwards per prompt: one captures the unsteered block output for its mean per-token norm, the
    second adds the vector and returns the steered logits."""
    last = model.n_layers - 1
    if alpha != 0 and layer % model.n_layers == last:
        raise ValueError(f"steer layer {layer} is the read-out layer; pick layer < {last}")
    if alpha == 0:
        return _final_logits(model, units)
    if scale_by == "token_norm":
        raise Unsupported("'add' carries one width-shaped value per hook; a strength proportional "
                          "to each token's own norm needs a different value at every position "
                          "(intervention_specs.validate_spec)")
    name = f"blocks.{layer}.hook_out"
    token_id = _resolve_token(model.tokenizer, target)
    direction = F.normalize(_weight(model, "unembedding")[token_id], dim=0)
    rows = []
    for prompt in units:
        _, cache = _run(model, prompt, [name], logits=False)
        stream_norm = cache[name][0].float().norm(dim=-1).mean().item()
        rows.append(_final_logits(model, [prompt],
                                  {name: {"op": "add", "value": alpha * stream_norm * direction}}))
    return torch.cat(rows)                                                   # [B, vocab]


@cell("gen_steering", family="llama", backend="transformer_lens")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); per-step interventions during decoding are not "
                      "expressible")


@cell("activation_patching", family="llama", backend="transformer_lens")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all"):
    """`positions="last"`: capture the clean block output, then `set` it at the corrupt run's last
    position (`pos`, which needs boot_vllm(enable_position_interventions=True))."""
    if not patch:
        return _final_logits(model, [unit[1] for unit in units])     # the corrupt runs, unpatched
    if positions != "last":
        raise Unsupported("an intervention carries one width-shaped value per hook; writing a "
                          "different clean activation at every position in one forward is not "
                          "expressible; a single-position patch is ('set' with 'pos')")
    name = f"blocks.{layer}.hook_out"
    rows = []
    for clean_prompt, corrupt_prompt in units:
        _, cache = _run(model, clean_prompt, [name], logits=False)
        clean_last = cache[name][0, -1, :].float()
        corrupt_len = _ids(model, corrupt_prompt).shape[-1]
        rows.append(_final_logits(model, [corrupt_prompt],
                                  {name: {"op": "set", "value": clean_last, "pos": corrupt_len - 1}}))
    return torch.cat(rows)                                                   # [B, vocab]


@cell("ablation", family="llama", backend="transformer_lens")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(model, units)
    return _final_logits(model, units, {f"blocks.{layer}.{target}.hook_out": {"op": "suppress"}})


@cell("activation_harvest", family="llama", backend="transformer_lens")
def activation_harvest(be, model, units, *, layer, export=True):
    """The compiled vLLM driver takes one prompt per forward (`driver.py`, `_normalize_input_ids`:
    batch_size=1 only), so the batch runs as one `run_with_cache` call per prompt. The worker copies
    the `blocks.{layer}.hook_out` buffer to host memory as wire bytes (`worker_extension.py`,
    `tl_read_captures`). `export=False` runs the same forwards with no capture and no logit
    rebuild, the backend's plain request, which returns no output."""
    name = f"blocks.{layer}.hook_out"
    caches = [_run(model, prompt, [name] if export else [], logits=False)[1] for prompt in units]
    return torch.cat([cache[name][0] for cache in caches]) if export else None   # [tokens, d_model]


@cell("gen_capture", family="llama", backend="transformer_lens")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); a read at every decode step is not expressible")

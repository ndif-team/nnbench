"""TransformerLens 4.0 vLLM-bridge cells with the readout moved off the CPU (backend
`transformer_lens_opt`).

Identical interventions to backends/transformer-lens-vllm/cells.py (they already run inside the
compiled graph), one forward per unit as there. What changes is the readout: the driver rebuilds
logits for every prompt position on the CPU in fp32 for any call that returns them (driver.py,
`_reconstruct_logits`), with no documented last-position or GPU option. Here every call runs with
`return_logits=False`, captures the final block's output, and applies the final RMSNorm and the
unembedding to the last row on the GPU with fp32 weights fetched once through the driver's
`get_param`. The logit lens projects its rows on the GPU the same way. Outputs carry the unit axis
at dim -2.
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
    """Each prompt's next-token logits from the last row of the final block's output, normed and
    unembedded on the GPU, one forward each: [B, vocab]."""
    name = f"blocks.{model.n_layers - 1}.hook_out"
    rows = [_run(model, prompt, [name], intervene=intervene, logits=False)[1][name][0, -1, :]
            for prompt in prompts]
    return _project(model, torch.stack(rows))


def _project(model, rows):
    """Final RMSNorm, then the unembedding, on the GPU in fp32. rows: [..., d_model]."""
    x = rows.to(model.device, torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + model.config.rms_norm_eps)
    x = x * _gpu_weight(model, "model.norm.weight")
    return (x @ _gpu_weight(model, "unembedding").T).cpu()


def _gpu_weight(model, name):
    key = f"gpu:{name}"
    if key not in model.weights:
        model.weights[key] = _weight(model, name).to(model.device)
    return model.weights[key]


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


@cell("logit_lens", family="llama", backend="transformer_lens_opt")
def logit_lens(be, model, units, *, layers="all"):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the GPU."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    names = [f"blocks.{i}.hook_out" for i in idx]
    rows = []
    for prompt in units:
        _, cache = _run(model, prompt, names, logits=False)
        rows.append(torch.stack([cache[name][0, -1, :] for name in names]))  # [n_layers, d_model]
    return _project(model, torch.stack(rows, dim=1))                         # [n_layers, B, vocab]


@cell("steering", family="llama", backend="transformer_lens_opt")
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


@cell("gen_steering", family="llama", backend="transformer_lens_opt")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); per-step interventions during decoding are not "
                      "expressible")


@cell("activation_patching", family="llama", backend="transformer_lens_opt")
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


@cell("ablation", family="llama", backend="transformer_lens_opt")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(model, units)
    return _final_logits(model, units, {f"blocks.{layer}.{target}.hook_out": {"op": "suppress"}})

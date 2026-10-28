"""TransformerLens 4.0 vLLM-bridge cells with the readout moved off the CPU (backend
`transformer_lens_opt`).

Identical interventions and forwards to backends/transformer-lens-vllm/cells.py, whose helpers
these cells call (one forward per unit on the compiled path, one per call on the batched path).
What changes is the readout: the driver rebuilds logits for every prompt position on the CPU in
fp32 for any call that returns them (driver.py, `_reconstruct_logits`), with no documented
last-position or GPU option. Here every call runs with `return_logits=False`, captures the final
block's output, and applies the final RMSNorm and the unembedding to the last row on the GPU with
fp32 weights fetched once through the driver's `get_param`. The logit lens and the Jacobian lens
project their rows on the GPU the same way. Outputs carry the unit axis at dim -2.
"""
import torch
import torch.nn.functional as F

from isb.methodologies.registry import Unsupported, cell
from isb.methodologies.steering import _resolve_token
from isb.tasks.lens_eval import locate

from .cells import (_forward, _ids, _jlens_maps, _last_rows, _steer_edits, _steer_layers,
                    _transported, _weight, activation_harvest, gen_capture, residual_readout)


def _final_logits(be, model, prompts, intervene=None):
    """Each prompt's next-token logits from the last row of the final block's output, normed and
    unembedded on the GPU: [B, vocab]."""
    rows = _last_rows(be, model, prompts, [f"blocks.{model.n_layers - 1}.hook_out"],
                      intervene=intervene)
    return _project(model, rows[0])


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


@cell("logit_lens", family="llama", backend="transformer_lens_opt")
def logit_lens(be, model, units, *, layers="all"):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the GPU."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _last_rows(be, model, units, [f"blocks.{i}.hook_out" for i in idx])   # [n_layers, B, d]
    return _project(model, rows)                                             # [n_layers, B, vocab]


@cell("jacobian_lens", family="llama", backend="transformer_lens_opt")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """The captured block output at each prompt's readout position (`locate`), each layer's
    transport applied on the GPU in fp32, then the final RMSNorm and the unembedding there."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else list(layers)
    positions = [locate(model.tokenizer, unit, position) for unit in units]
    rows = _last_rows(be, model, units, [f"blocks.{i % n}.hook_out" for i in idx], positions)
    rows = rows.to(model.device, torch.float32)
    rows = _transported(rows, _jlens_maps(model, transport, rows.shape[-1], model.device), idx)
    return _project(model, rows)                                             # [n_layers, B, vocab]


cell("residual_readout", family="llama", backend="transformer_lens_opt")(residual_readout)
cell("activation_harvest", family="llama", backend="transformer_lens_opt")(activation_harvest)
cell("gen_capture", family="llama", backend="transformer_lens_opt")(gen_capture)


@cell("steering", family="llama", backend="transformer_lens_opt")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm"):
    """`add` takes a fixed value, so a strength relative to the stream's own norm needs a capture
    forward per steered layer before the steered forward (cells.py, `_steer_edits`). `layers`
    steers every listed layer in one forward and replaces `layer`."""
    sites = _steer_layers(model, layer, layers, alpha)
    if alpha == 0:
        return _final_logits(be, model, units)
    if scale_by == "token_norm":
        raise Unsupported("'add' carries one width-shaped value per hook; a strength proportional "
                          "to each token's own norm needs a different value at every position "
                          "(intervention_specs.validate_spec)")
    token_id = _resolve_token(model.tokenizer, target)
    direction = F.normalize(_weight(model, "unembedding")[token_id], dim=0)
    edits = _steer_edits(be, model, units, sites, direction, alpha)
    return torch.cat([_final_logits(be, model, [prompt], edit)
                      for prompt, edit in zip(units, edits)])                # [B, vocab]


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
        return _final_logits(be, model, [unit[1] for unit in units])   # the corrupt runs, unpatched
    if positions != "last":
        raise Unsupported("an intervention carries one width-shaped value per hook; writing a "
                          "different clean activation at every position in one forward is not "
                          "expressible; a single-position patch is ('set' with 'pos')")
    if be.batching:
        raise Unsupported("the eager batched path applies interventions to the whole tensor and "
                          "takes no 'pos' (driver.py, _validate_interventions; boot_vllm refuses "
                          "enable_position_interventions with enable_batching, source.py)")
    name = f"blocks.{layer}.hook_out"
    rows = []
    for clean_prompt, corrupt_prompt in units:
        (_, cache), = _forward(be, model, [clean_prompt], [name], logits=False)
        clean_last = cache[name][-1].float()
        corrupt_len = len(_ids(model, corrupt_prompt))
        rows.append(_final_logits(be, model, [corrupt_prompt],
                                  {name: {"op": "set", "value": clean_last, "pos": corrupt_len - 1}}))
    return torch.cat(rows)                                                   # [B, vocab]


@cell("ablation", family="llama", backend="transformer_lens_opt")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(be, model, units)
    return _final_logits(be, model, units, {f"blocks.{layer}.{target}.hook_out": {"op": "suppress"}})

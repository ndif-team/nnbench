"""TransformerLens 4.0 vLLM-bridge cells with the readout moved off the CPU (backend
`transformer_lens_opt`).

Identical interventions and forwards to backends/transformer-lens-vllm/cells.py, whose helpers
these cells call (one forward per unit on the compiled path, one per call on the batched path).
What changes is the readout: the driver rebuilds logits for every prompt position on the CPU in
fp32 for any call that returns them (driver.py, `_reconstruct_logits`), with no documented
last-position or GPU option. Here a call that needs logits runs with `return_logits=False`,
captures the final block's output, and applies the final RMSNorm and the unembedding to the last
row on the GPU with fp32 weights fetched once through the driver's `get_param`. The logit lens and
the Jacobian lens project their rows on the GPU the same way. A timed call that returns a token
(steering, ablation, activation patching, the activation-harvest baseline) reads the token vLLM
generated off the driver's forward result (`_next_tokens`) and rebuilds no logits. Outputs carry the
unit axis at dim -2.
"""
import torch

from isb.methodologies.registry import Unsupported, cell
from isb.data import locate

from .cells import (_ids, _jlens_maps, _last_rows, _patch_requests, _steer_requests, _transported,
                    _weight, _wire, activation_harvest, gen_capture, residual_readout)


def _next_tokens(be, model, prompts, intervene=None):
    """Each prompt's greedy next token, `[B, 1]` int64, from one request per prompt on the compiled
    path and one for all of them on the batched path, as `_forward` makes them. The driver's
    request (transformer_lens/model_bridge/sources/vllm/driver.py, `VLLMDriver.forward`) pushes the
    specs, runs `llm.generate` with `max_tokens=1, temperature=0.0` and returns vLLM's
    `RequestOutput` as `ForwardResult.raw_output` (driver_protocol.py); with `return_logits=False`
    it reads nothing back and rebuilds no logits. `RemoteBridge.forward` drops `raw_output`, so the
    request goes to the driver through the bridge's private `_driver`."""
    driver = model.bridge._driver
    intervene = _wire(be, intervene)
    rows = [_ids(model, prompt) for prompt in prompts]
    if be.batching:
        outputs = driver.forward(rows, intervene=intervene, return_logits=False).raw_output
    else:
        outputs = [driver.forward(torch.tensor([row]), intervene=intervene,
                                  return_logits=False).raw_output for row in rows]
    return torch.tensor([list(output.outputs[0].token_ids) for output in outputs])


def _final_logits(be, model, prompts, intervene=None):
    """Each prompt's next-token logits from the last row of the final block's output, normed and
    unembedded on the GPU: [B, vocab]."""
    rows = _last_rows(be, model, prompts, [f"blocks.{model.n_layers - 1}.hook_out"],
                      intervene=intervene)
    return _project(model, rows[0], keep=False)


def _project(model, rows, keep=True, top=False):
    """Final RMSNorm, then the unembedding, on the GPU in fp32, or with `top` the top-1 token ids
    `[..., 1]`. rows: [..., d_model]. `keep` as `_gpu_weight`."""
    x = rows.to(model.device, torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + model.config.rms_norm_eps)
    x = x * _gpu_weight(model, "model.norm.weight", keep)
    logits = x @ _gpu_weight(model, "unembedding", keep).T
    return (logits.argmax(-1, keepdim=True) if top else logits).cpu()


def _gpu_weight(model, name, keep):
    """A weight (`_weight`) on the GPU: kept for later calls when `keep` (the lens readouts a
    timed call runs), else for this call only, so the GPU holds no weights that only check calls
    use."""
    key = f"gpu:{name}"
    if key in model.weights:
        return model.weights[key]
    weight = _weight(model, name).to(model.device)
    if keep:
        model.weights[key] = weight
    return weight


@cell("logit_lens", family="llama", backend="transformer_lens_opt")
def logit_lens(be, model, units, *, layers="all", check=False):
    """Capture every block's output stream, then apply the final RMSNorm and the unembedding on
    the GPU: each layer's top-1 token id `[n_layers, B, 1]` for a timed call, the logits
    `[n_layers, B, vocab]` for a check call."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _last_rows(be, model, units, [f"blocks.{i}.hook_out" for i in idx])   # [n_layers, B, d]
    return _project(model, rows, top=not check)


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
cell("gen_capture", family="llama", backend="transformer_lens_opt")(gen_capture)


@cell("activation_harvest", family="llama", backend="transformer_lens_opt")
def activation_harvest_opt(be, model, units, *, layer=None, layers=None, export=True):
    """The general cell's capture forwards (cells.py, `activation_harvest`); the `export=False`
    baseline returns the token each request generated, `[B, 1]` (`_next_tokens`)."""
    if not export:
        return _next_tokens(be, model, units)
    return activation_harvest(be, model, units, layer=layer, layers=layers)


@cell("steering", family="llama", backend="transformer_lens_opt")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm", check=False):
    """The general cell's steered requests (cells.py, `_steer_requests`): capture forwards first
    under `mean_norm`, fixed values measured once per loaded engine under
    `calibrated_mean_norm`, no spec at `alpha=0`. A timed call returns the token each request
    generated, `[B, 1]` (`_next_tokens`); a check call returns the next-token logits projected on
    the GPU, `[B, vocab]`."""
    requests = _steer_requests(be, model, units, layer, layers, target, alpha, scale_by)
    read = _final_logits if check else _next_tokens
    return torch.cat([read(be, model, prompts, edit) for prompts, edit in requests])


@cell("gen_steering", family="llama", backend="transformer_lens_opt")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported("the vLLM driver runs one forward per call (max_new_tokens=1 only) and the "
                      "bridge has no generate(); per-step interventions during decoding are not "
                      "expressible")


@cell("activation_patching", family="llama", backend="transformer_lens_opt")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all", check=False):
    """The general cell's patched requests (cells.py, `_patch_requests`). A timed call returns the
    token each request generated, `[B, 1]` (`_next_tokens`); a check call returns the next-token
    logits projected on the GPU, `[B, vocab]`."""
    requests = _patch_requests(be, model, units, layer, patch, positions)
    read = _final_logits if check else _next_tokens
    return torch.cat([read(be, model, prompts, edit) for prompts, edit in requests])


@cell("ablation", family="llama", backend="transformer_lens_opt")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """`suppress` on `blocks.{layer}.{target}.hook_out`, as the general cell sends it; no spec at
    `target="none"`. A timed call returns the token each request generated, `[B, 1]`
    (`_next_tokens`); a check call returns the next-token logits projected on the GPU, `[B, vocab]`."""
    site = f"blocks.{layer}.{target}.hook_out"
    intervene = None if target == "none" else {site: {"op": "suppress"}}
    read = _final_logits if check else _next_tokens
    return read(be, model, units, intervene)

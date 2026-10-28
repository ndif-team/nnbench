"""vLLM-Hook cells for the cross-system workloads, built-in workers only.

Denotation, from IBM/vLLM-Hook 0e34fdd source:
- `probe_hidden_states` hooks every decoder layer and, on vLLM's fused-residual layers, records
  `output[0] + output[1]`, the stream after the layer (`probe_hidden_states_worker.hs_hook`), per
  request from its `query_start_loc` slice. Layers are numbered from 1 ("layer N = output after
  the Nth block"); `last_token` keeps one row per forward, `all_tokens` the request's whole slice.
  Capture is prefill-only by default.
- `probe_hook_qk` hooks each attention module `model.layers.<i>.self_attn.attn` (numbered from 0,
  `_common.match_attn`) and records its inputs: the query and key after vLLM applied RoPE
  (vLLM 0.19.1 `qwen2.py`, `Qwen2Attention.forward`: `q, k = self.rotary_emb(positions, q, k)`
  then `self.attn(q, k, v)`), the query at the last token (`hookq_mode` "last_token") and the keys
  of every token. Its `attn_tracker` analyzer rebuilds the attention probabilities off the kernel
  (`compute_attention_from_qk`: `softmax(q @ k.T * attention_multiplier)` per head, the KV heads
  repeated for grouped-query attention).
- A call's units go out as one `HookLLM.generate` over all their prompts. On the RPC path the
  plugin merges the requests' probes onto `outputs[0].probes` as a list in request order
  (`hook_llm.py`, `generate`); on the disk path `flush_disk` writes the batch's requests into one
  artifact in request order. The `hidden_states` analyzer returns those rows per layer.
- The plugin returns hidden states only, with no unembedding. The next-token readout is user code:
  the checkpoint's final RMSNorm and `lm_head` weights, applied to the last rows on the GPU.
- `steer_hook_act`, the steering worker, adds a fixed vector (`add_vector`) or adjusts a projection
  (`adjust_rs`) at the LAST row of each forward of one layer: the last prompt token, and each decoded
  token when `apply_at_all_positions` is true (commit afdf36a). It is a separate worker extension,
  so an engine either reads or steers.
Workloads outside those forms are declared unsupported with the reason. Outputs carry the unit
axis at dim -2.
"""
import json
import os
import shutil

import torch

from isb.methodologies.jacobian_lens import _load_fitted, _orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.tasks.lens_eval import locate

ONE_ENGINE = ("an engine carries one worker extension (HookLLM worker_name), so reading the stream "
              "(probe_hidden_states) and writing it (steer_hook_act) cannot share an engine")
LAST_ROW_STEER = ("steer_hook_act writes only the last row of each forward "
                  "(steer_activation_worker._apply_steer: slice_view[-1:]), the last prompt token and "
                  "each decoded token; adding at every prompt position is not expressible")


def _other_worker(model, needed, what):
    return Unsupported(f"{what} is the {needed} worker, and this engine carries {model.worker}: an "
                       f"engine carries one worker extension (HookLLM worker_name)")


def _weights(model):
    """The final norm and unembedding from the checkpoint, on the GPU in fp32, loaded once."""
    if not model.weights:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        root = snapshot_download(model.repo, local_files_only=True)
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index_path):                       # a sharded checkpoint
            index = json.load(open(index_path))["weight_map"]
        else:                                                # a single-file checkpoint
            with safe_open(os.path.join(root, "model.safetensors"), framework="pt") as f:
                index = {name: "model.safetensors" for name in f.keys()}
        names = {"norm": "model.norm.weight",
                 "unembed": "lm_head.weight" if "lm_head.weight" in index else "model.embed_tokens.weight"}
        for key, name in names.items():
            with safe_open(os.path.join(root, index[name]), framework="pt", device="cuda") as f:
                model.weights[key] = f.get_tensor(name).float()
    return model.weights


def _project(model, rows):
    """Final RMSNorm, then the unembedding, on the GPU in fp32. rows: [n, d_model] -> [n, vocab]."""
    w = _weights(model)
    x = rows.to("cuda", torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + model.config.rms_norm_eps) * w["norm"]
    return (x @ w["unembed"].T).cpu()


def _requests(tensors, n):
    """One layer's captured passes of each of `n` requests, in prompt order, from either artifact
    format: the RPC path stacks one request's passes in a tensor and lists several requests' first
    passes; the disk path lists every request's passes. A last-token pass is `[d]` (or `[1, d]`),
    an all-tokens pass `[n_tokens, d]`."""
    rows = list(tensors.unbind(0)) if isinstance(tensors, torch.Tensor) else list(tensors)
    if len(rows) != n:
        raise RuntimeError(f"collected {len(rows)} captured passes, expected {n}")
    return rows


def _capture(model, prompts, layers, mode):
    """One `HookLLM.generate` over `prompts` capturing the stream after each listed (0-based) layer
    in hs mode `mode`; returns the outputs and the analyzer's per-layer captures."""
    from vllm import SamplingParams

    if model.worker != "probe_hidden_states":
        raise _other_worker(model, "probe_hidden_states", "hidden-state capture")
    llm = model.llm
    llm._output_layers = [i + 1 for i in layers]          # the worker numbers layers from 1
    previous, llm._hs_mode = llm._hs_mode, mode           # what load_config sets from a config file
    try:
        outs = llm.generate(list(prompts), SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0),
                            save_to_disk=model.save_to_disk, use_tqdm=False)
        if model.save_to_disk:
            stats = llm.analyze(analyzer_spec={"reduce": "none"})   # reads this call's run_id
            shutil.rmtree(os.path.join(llm._hook_dir, llm._last_run_id), ignore_errors=True)
        else:
            stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=outs[0].probes)
    finally:
        llm._hs_mode = previous
    return outs, stats["hidden_states"]


def _last_rows(model, prompts, layers):
    """Each listed layer's (0-based) last-token stream of every prompt: [len(layers), B, d_model]."""
    _, by_layer = _capture(model, prompts, layers, "last_token")
    return torch.stack([torch.stack([row.reshape(-1) for row in
                                     _requests(by_layer[f"model.layers.{i}"], len(prompts))])
                        for i in layers])


def _token_rows(model, prompts, layers):
    """Each listed layer's (0-based) stream at every token of every prompt: per layer, the
    prompts' `[n_tokens, d_model]` rows in prompt order. A prompt whose rows fall short of its
    token count was split over several forwards (a chunked prefill), whose first pass the RPC
    merge keeps alone (`hook_llm.py`, `generate`)."""
    outs, by_layer = _capture(model, prompts, layers, "all_tokens")
    counts = [len(out.prompt_token_ids) for out in outs]
    per_layer = []
    for i in layers:
        rows = _requests(by_layer[f"model.layers.{i}"], len(prompts))
        if [row.shape[0] for row in rows] != counts:
            raise RuntimeError(f"layer {i} captured {[row.shape[0] for row in rows]} rows per "
                               f"prompt, expected the prompt lengths {counts}")
        per_layer.append(rows)
    return per_layer


def _final_logits(model, prompts):
    return _project(model, _last_rows(model, prompts, [model.n_layers - 1])[0])   # [B, vocab]


@cell("logit_lens", family="llama", backend="vllm_hook")
def logit_lens(be, model, units, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _last_rows(model, units, idx)                                       # [n_layers, B, d]
    logits = _project(model, rows.reshape(-1, rows.shape[-1]))
    return logits.reshape(len(idx), len(units), -1)                            # [n_layers, B, vocab]


@cell("steering", family="llama", backend="vllm_hook")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm"):
    if alpha == 0:
        return _final_logits(model, units)
    raise Unsupported(f"{LAST_ROW_STEER}; a strength from the stream's norm also needs a read, and "
                      f"{ONE_ENGINE}")


@cell("gen_steering", family="llama", backend="vllm_hook")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported(f"{LAST_ROW_STEER}; steering at every decode step steers the last prompt "
                      f"token, not every prompt position")


@cell("activation_patching", family="llama", backend="vllm_hook")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all"):
    if not patch:
        return _final_logits(model, [unit[1] for unit in units])   # the corrupt runs, unpatched
    if positions != "last":
        raise Unsupported(LAST_ROW_STEER)
    raise Unsupported("a last-token patch is an add_vector of clean minus corrupt at the last row, "
                      f"but both values must be read first, and {ONE_ENGINE}")


@cell("ablation", family="llama", backend="vllm_hook")
def ablation(be, model, units, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(model, units)
    raise Unsupported("the built-in workers hook decoder layers and attention modules for reads and "
                      "steer the residual by a fixed vector; none zeroes a submodule's output")


def _transport(model, transport, layers, dtype):
    """The J-lens transport for the read `layers` on the GPU in the stream's `dtype`: None (the
    identity), one `[d, d]` map shared by every layer (an int seed, `_orthogonal`, or a tensor), or
    an `[n_layers, d, d]` stack of per-layer maps (a fitted lens, `_load_fitted`, or a mapping).
    A seed or a fitted spec is built once per model and kept."""
    if transport is None:
        return None
    key = (transport, tuple(layers), dtype) if isinstance(transport, (int, str)) else None
    if key in model.transports:
        return model.transports[key]
    if isinstance(transport, int):
        maps = _orthogonal(model.config.hidden_size, transport)
    elif isinstance(transport, torch.Tensor):
        maps = transport
    else:
        per_layer = _load_fitted(transport) if isinstance(transport, str) else transport
        maps = torch.stack([per_layer[i] for i in layers])
    maps = maps.to("cuda", dtype)
    if key is not None:
        model.transports[key] = maps
    return maps


@cell("jacobian_lens", family="llama", backend="vllm_hook")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """The J-lens readout from the plugin's hidden states: the stream after each read layer at
    each prompt's readout position (`last_token` capture for the last token, `all_tokens` for any
    other), then user code on the client GPU, as for the logit lens: `J_l @ h` in the stream's
    dtype, the checkpoint's final RMSNorm and the unembedding (`_project`, fp32). The plugin has
    no unembedding, so the projection cannot run in the worker. Output `[n_layers, B, vocab]`."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    positions = [locate(model.tokenizer, unit, position) for unit in units]
    if all(pos == -1 for pos in positions):
        rows = _last_rows(model, units, idx)                                     # [n_layers, B, d]
    else:
        rows = torch.stack([torch.stack([prompt[pos] for prompt, pos in zip(per_prompt, positions)])
                            for per_prompt in _token_rows(model, units, idx)])
    x = rows.to("cuda")
    maps = _transport(model, transport, idx, x.dtype)
    if maps is not None:
        x = x @ maps.T if maps.dim() == 2 else x @ maps.transpose(1, 2)        # per layer: h @ J_l.T
    logits = _project(model, x.reshape(-1, x.shape[-1]))
    return logits.reshape(len(idx), len(units), -1)                            # [n_layers, B, vocab]


@cell("activation_harvest", family="llama", backend="vllm_hook")
def activation_harvest(be, model, units, *, layer=None, layers=None, export=True):
    """The plugin's `all_tokens` capture of the stream after the layer, one `HookLLM.generate`
    for the whole batch, each prompt's rows concatenated in prompt order: `[tokens, d]`, or
    `[len(layers), tokens, d]` for `layers` (then `layer` is ignored). The engine's hs mode is
    restored after the call. `export=False` runs the same generate without the hook
    (`use_hook=False`) and returns the greedy tokens."""
    from vllm import SamplingParams

    if not export:
        outs = model.llm.generate(list(units), SamplingParams(max_tokens=1, temperature=0.0,
                                                              top_p=1.0),
                                  use_hook=False, use_tqdm=False)
        return torch.tensor([[out.outputs[0].token_ids[0]] for out in outs])      # [B, 1]
    n = model.n_layers
    read = [i % n for i in (layers if layers is not None else [layer])]
    stacked = torch.stack([torch.cat(rows) for rows in _token_rows(model, units, read)])
    return stacked if layers is not None else stacked[0]


@cell("residual_readout", family="llama", backend="vllm_hook")
def residual_readout(be, model, units, *, layers):
    """The stream after each listed layer at each prompt's last token, the plugin's `last_token`
    capture: `[len(layers), B, d]` in the engine's dtype, on the host."""
    n = model.n_layers
    return _last_rows(model, units, [i % n for i in layers])


@cell("attention_pattern", family="llama", backend="vllm_hook")
def attention_pattern(be, model, units, *, layers="all"):
    """The last query's attention log-probabilities `[n_layers, B, n_heads, k_len]` from the Q/K
    probe: per request, `output_qk` lists the read layers (the worker numbers attention modules
    from 0), the worker records the post-RoPE query at the last token and the keys of the whole
    prompt, and the `attn_tracker` analyzer's own reconstruction gives each head's probabilities
    over the prompt (`compute_attention_from_qk`, GQA heads repeated); the log is user code."""
    from vllm import SamplingParams

    if model.worker != "probe_hook_qk":
        raise _other_worker(model, "probe_hook_qk", "Q/K capture")
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    llm = model.llm
    params = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, extra_args={"output_qk": idx})
    outs = llm.generate(list(units), params, save_to_disk=model.save_to_disk, use_tqdm=False)
    if model.save_to_disk:
        weights = llm.analyzer.compute_attention_from_qk(run_id=llm._last_run_id)
        shutil.rmtree(os.path.join(llm._hook_dir, llm._last_run_id), ignore_errors=True)
    else:
        weights = llm.analyzer.compute_attention_from_qk(probes=outs[0].probes)
    if len(weights) != len(units):
        raise RuntimeError(f"reconstructed {len(weights)} requests' attention, expected {len(units)}")
    return torch.stack([torch.stack([torch.log(w[f"model.layers.{i}.self_attn.attn"]["attention"])
                                     for w in weights]) for i in idx])

"""vLLM-Hook cells for the cross-system workloads, built-in workers only.

Denotation, from IBM/vLLM-Hook 0e34fdd source:
- `probe_hidden_states` hooks every decoder layer and, on vLLM's fused-residual layers, records
  `output[0] + output[1]`, the stream after the layer (`probe_hidden_states_worker.hs_hook`), per
  request from its `query_start_loc` slice. Layers are numbered from 1 ("layer N = output after
  the Nth block"); `last_token` keeps one row per forward, `all_tokens` the request's whole slice.
  Capture is prefill-only by default; a request's `extra_args["hooks_on"] = "both"` adds every
  decode forward (`hs_hook`), the last-token row of each in last-token mode.
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
- The plugin returns hidden states only, with no unembedding. A readout over the vocabulary is user
  code: the checkpoint's final RMSNorm and `lm_head` weights applied to captured rows on the GPU in
  fp32, as vLLM-Hook's analyzers replay the norm on fp32 features (`ScienceHallucinationAnalyzer`).
- `steer_hook_act`, the steering worker, adds a fixed vector (`add_vector`) or adjusts a projection
  (`adjust_rs`) at the LAST row of each forward of one layer: the last prompt token, and each decoded
  token when `apply_at_all_positions` is true (commit afdf36a). It is a separate worker extension,
  so an engine either reads or steers.
Workloads outside those forms are declared unsupported with the reason. Outputs carry the unit
axis at dim -2.

MIA (v1.0.1, d748ccf, on vLLM 0.32.0) keeps those read denotations and the same calls on `MiaLLM`:
- `capture_hs` records `hidden + residual` after each decoder block, in the CUDA graph
  (`mia/core/hooks/ops.py`, the capture op) and in the eager hook
  (`mia/workers/hs_capture_worker.py`, `hs_hook`: `output[0] + output[1]`), layers numbered from 1,
  the same `last_token` / `all_tokens` modes and `hooks_on` passes. A request names its layers and
  mode in its own `extra_args` (`output_hidden_states`, `hs_mode`; docs/configs.md, "Per-request
  arguments"), which `MiaLLM.generate` keeps over the engine's defaults.
- `capture_qk` records the same post-RoPE query and keys, and `attn_tracker` rebuilds the
  probabilities (`compute_attention_from_qk`).
- A batch's probes arrive merged on `outputs[0].probes`, one first pass per request
  (`mia/llm.py`, `_merge_probes`); `save_to_disk` writes the run under the given `run_id`.
- `steer` adds `coefficient * dir` at every token of the listed blocks, or the last one
  (`steering.positions`), on the prefill and decode passes (`steering.phase`); it writes no
  artifact and returns no activations (docs/configs.md).
"""
import json
import os
import uuid

import torch

from isb.methodologies.jacobian_lens import load_fitted, orthogonal
from isb.methodologies.registry import Unsupported, cell
from isb.data import locate

from .backend import MIA, QK, READ, AsyncHook

ONE_ENGINE = ("an engine carries one worker extension (HookLLM worker_name), so reading the stream "
              "(probe_hidden_states) and writing it (steer_hook_act) cannot share an engine")
LAST_ROW_STEER = ("steer_hook_act writes only the last row of each forward "
                  "(steer_activation_worker._apply_steer: slice_view[-1:]), the last prompt token and "
                  "each decoded token; adding at every prompt position is not expressible")
# MIA's steer worker reaches every token, and the engine that steers still returns nothing to read.
MIA_STEER = ("the steer worker adds a fixed vector (steering.method add_vector: coefficient * dir "
             "at the listed blocks) and returns no activations (docs/configs.md: steering writes "
             "no artifact), and an engine carries one worker (MiaLLM worker_name), so the engine "
             "that steers cannot read the stream: neither the norm the strength is relative to "
             "nor the steered next-token distribution over the vocabulary")


def _other_worker(model, needed, what):
    return Unsupported(f"{what} is the {needed} worker, and this engine carries {model.worker}: an "
                       f"engine carries one worker extension (HookLLM worker_name)")


def _stored_weights(model):
    """The checkpoint's final norm and unembedding in host memory as stored, read once."""
    if "host" not in model.weights:
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
        weights = {}
        for key, name in names.items():
            with safe_open(os.path.join(root, index[name]), framework="pt") as f:
                weights[key] = f.get_tensor(name)
        model.weights["host"] = weights
    return model.weights["host"]


def _weights(model, keep):
    """The final norm and unembedding on the GPU in fp32: kept for later calls when `keep` (the
    readout a timed call runs), else for this call only, so the GPU holds no weights that only check
    calls use."""
    if "gpu" in model.weights:
        return model.weights["gpu"]
    weights = {key: w.to("cuda", torch.float32) for key, w in _stored_weights(model).items()}
    if keep:
        model.weights["gpu"] = weights
    return weights


def _project(model, rows, keep):
    """Final RMSNorm, then the unembedding, on the GPU in fp32; the result on the host.
    rows: [n, d_model] -> [n, vocab]. `keep` as `_weights`."""
    w = _weights(model, keep)
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


def _capture(model, prompts, layers, mode, new_tokens=1):
    """One `HookLLM.generate` over `prompts` capturing the stream after each listed (0-based) layer
    in hs mode `mode`; returns the outputs and the analyzer's per-layer captures. With `new_tokens`
    > 1 each prompt decodes that many greedy tokens and the requests ask for the decode passes too
    (`extra_args["hooks_on"] = "both"`, probe_hidden_states_worker; generate keeps a caller's
    extra_args and adds its own defaults)."""
    from vllm import SamplingParams

    if model.worker != READ:
        raise _other_worker(model, READ, "hidden-state capture")
    if isinstance(model.llm, AsyncHook):           # AsyncLLM: the fields go on each request
        return model.llm.capture(prompts, [i + 1 for i in layers], mode, new_tokens)
    if MIA:
        return _capture_mia(model, prompts, layers, mode, new_tokens)
    params = (SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0, top_p=1.0,
                             extra_args={"hooks_on": "both"}) if new_tokens > 1
              else SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0))
    llm = model.llm
    llm._output_layers = [i + 1 for i in layers]          # the worker numbers layers from 1
    previous, llm._hs_mode = llm._hs_mode, mode           # what load_config sets from a config file
    try:
        outs = llm.generate(list(prompts), params, save_to_disk=model.save_to_disk, use_tqdm=False)
        if model.save_to_disk:
            stats = llm.analyze(analyzer_spec={"reduce": "none"})   # reads this call's run_id
            model.finished_runs.append(os.path.join(llm._hook_dir, llm._last_run_id))
        else:
            stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=outs[0].probes)
    finally:
        llm._hs_mode = previous
    return outs, stats["hidden_states"]


def _capture_mia(model, prompts, layers, mode, new_tokens):
    """`_capture` on MIA: every request carries its layers (numbered from 1), mode and passes in
    its own `extra_args`, and a disk run is named by this call, read back and left for `between_calls` to remove.
    `probes` is
    read inside the call: MIA delivers an output's capture on first access (examples/README.md,
    "Where the captured data goes")."""
    from vllm import SamplingParams

    extra = {"output_hidden_states": [i + 1 for i in layers], "hs_mode": mode}
    if new_tokens > 1:
        extra["hooks_on"] = "both"
    params = SamplingParams(max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0,
                            top_p=1.0, extra_args=extra)
    llm = model.llm
    if not model.save_to_disk:
        outs = llm.generate(list(prompts), params, use_tqdm=False)
        return outs, llm.analyze(analyzer_spec={"reduce": "none"},
                                 probes=outs[0].probes)["hidden_states"]
    run_id = uuid.uuid4().hex
    outs = llm.generate(list(prompts), params, save_to_disk=True, run_id=run_id, use_tqdm=False)
    stats = llm.analyze(analyzer_spec={"reduce": "none"}, run_id=run_id)
    model.finished_runs.append(os.path.join(model.hook_dir, run_id))
    return outs, stats["hidden_states"]


def _last_rows(model, prompts, layers):
    """Each listed layer's (0-based) last-token stream of every prompt: [len(layers), B, d_model]."""
    _, by_layer = _capture(model, prompts, layers, "last_token")
    rows = [row.reshape(-1) for i in layers
            for row in _requests(by_layer[f"model.layers.{i}"], len(prompts))]
    return torch.stack(rows).reshape(len(layers), len(prompts), -1)


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
    """The check form of a next token: the stream after the last layer projected in fp32,
    `[B, vocab]`."""
    return _project(model, _last_rows(model, prompts, [model.n_layers - 1])[0], keep=False)


def _plain_tokens(model, prompts):
    """Each prompt's greedy next token among the non-stop tokens (`min_tokens=1`), `[B, 1]` int64,
    from the plain request: `generate` with `use_hook=False`, which clears the requests'
    extra_args so the worker captures nothing (`vllm_hook_plugins/hook_llm.py` and `mia/llm.py`,
    `generate`), or on AsyncLLM one request per prompt with no extra_args (`AsyncHook.decode`).
    vLLM's request outputs come back in prompt order."""
    from vllm import SamplingParams

    if isinstance(model.llm, AsyncHook):
        outs = model.llm.decode(list(prompts), 1)
    else:
        outs = model.llm.generate(list(prompts), SamplingParams(
            max_tokens=1, min_tokens=1, temperature=0.0, top_p=1.0), use_hook=False, use_tqdm=False)
    return torch.tensor([list(out.outputs[0].token_ids) for out in outs])


@cell("logit_lens", family="llama", backend="vllm_hook")
def logit_lens(be, model, units, *, layers="all"):
    """The final RMSNorm and unembedding applied in fp32 to the stream after each read layer at
    each prompt's last token, `[n_layers, B, vocab]` on the host (`_project`)."""
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    rows = _last_rows(model, units, idx)                                       # [n_layers, B, d]
    logits = _project(model, rows.reshape(-1, rows.shape[-1]), keep=True)
    return logits.reshape(len(idx), len(units), -1)                            # [n_layers, B, vocab]


@cell("steering", family="llama", backend="vllm_hook")
def steering(be, model, units, *, layer=8, layers=None, target=" Rome", alpha=6.0,
             scale_by="mean_norm", check=False):
    """At `alpha=0` nothing is written: a timed call is the plain request and returns each unit's
    greedy token, `[B, 1]` (`_plain_tokens`); a check call projects the stream captured after the
    last layer, `[B, vocab]` (`_final_logits`). Any other strength is unsupported in both calls."""
    if alpha == 0:
        return _final_logits(model, units) if check else _plain_tokens(model, units)
    if MIA:
        raise Unsupported(MIA_STEER)
    raise Unsupported(f"{LAST_ROW_STEER}; a strength from the stream's norm also needs a read, and "
                      f"{ONE_ENGINE}")


@cell("gen_steering", family="llama", backend="vllm_hook")
def gen_steering(be, model, units, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    if MIA:
        raise Unsupported(MIA_STEER)
    raise Unsupported(f"{LAST_ROW_STEER}; steering at every decode step steers the last prompt "
                      f"token, not every prompt position")


@cell("gen_capture", family="llama", backend="vllm_hook")
def gen_capture(be, model, units, *, layer, new_tokens, export=True):
    """The stream after `layer` at each greedy decode step's newest position. In last-token mode the
    worker keeps the final prefill chunk's last row, then one row per decode forward, so each
    request yields `new_tokens` passes in step order; the requests' passes come back in prompt
    order (AsyncHook.capture reads one request at a time; `flush_disk` writes a batch's requests
    one after another). `export=False` runs the same decode with no hook fields and returns each
    step's token."""
    from vllm import SamplingParams

    if not export:
        if isinstance(model.llm, AsyncHook):
            outs = model.llm.decode(list(units), new_tokens)
        else:
            outs = model.llm.generate(list(units), SamplingParams(
                max_tokens=new_tokens, min_tokens=new_tokens, temperature=0.0, top_p=1.0),
                use_hook=False, use_tqdm=False)
        return torch.stack([torch.tensor(out.outputs[0].token_ids)[:, None] for out in outs],
                           dim=1)                                               # [new_tokens, B, 1]
    if not model.save_to_disk and not isinstance(model.llm, AsyncHook) and len(units) > 1:
        if MIA:
            raise Unsupported("MiaLLM.generate merges several requests' probes onto outputs[0], "
                              "keeping each request's first pass (mia/llm.py, _merge_probes), so a "
                              "batch's decode passes are not on it; disk storage keeps them")
        raise Unsupported("HookLLM.generate merges several requests' RPC probes onto outputs[0], "
                          "keeping each request's first pass (hook_llm.py, generate), so a batch's "
                          "decode passes are lost; disk storage and the AsyncLLM path keep them")
    _, by_layer = _capture(model, units, [layer], "last_token", new_tokens)
    captured = by_layer[f"model.layers.{layer}"]
    passes = captured.unbind(0) if isinstance(captured, torch.Tensor) else captured
    rows = torch.cat([p.reshape(-1, p.shape[-1]) for p in passes])
    if rows.shape[0] != len(units) * new_tokens:
        raise RuntimeError(f"collected {rows.shape[0]} per-step rows, expected {len(units)} "
                           f"requests x {new_tokens} steps")
    return rows.reshape(len(units), new_tokens, -1).transpose(0, 1)             # [new_tokens, B, d]


@cell("activation_patching", family="llama", backend="vllm_hook")
def activation_patching(be, model, units, *, layer=6, patch=True, positions="all", check=False):
    """`patch=False` runs each pair's corrupt prompt unpatched: a timed call is the plain request
    and returns its greedy token, `[B, 1]` (`_plain_tokens`); a check call projects the stream
    captured after the last layer, `[B, vocab]` (`_final_logits`). A patch is unsupported."""
    if not patch:
        corrupt = [unit[1] for unit in units]
        return _final_logits(model, corrupt) if check else _plain_tokens(model, corrupt)
    if MIA:
        raise Unsupported("a patch writes a value read from another run: the steer worker adds one "
                          "vector from a file to the positions it steers (steering.vector_path), "
                          "and an engine carries one worker (MiaLLM worker_name), so the engine "
                          "that steers reads neither the clean stream nor the patched output")
    if positions != "last":
        raise Unsupported(LAST_ROW_STEER)
    raise Unsupported("a last-token patch is an add_vector of clean minus corrupt at the last row, "
                      f"but both values must be read first, and {ONE_ENGINE}")


@cell("ablation", family="llama", backend="vllm_hook")
def ablation(be, model, units, *, layer=6, target="mlp", check=False):
    """At `target="none"` nothing is zeroed: a timed call is the plain request and returns each
    unit's greedy token, `[B, 1]` (`_plain_tokens`); a check call projects the stream captured
    after the last layer, `[B, vocab]` (`_final_logits`). Any other target is unsupported."""
    if target == "none":
        return _final_logits(model, units) if check else _plain_tokens(model, units)
    raise Unsupported("the built-in workers hook decoder layers and attention modules for reads and "
                      "steer the residual by a fixed vector; none zeroes a submodule's output")


def _transport(model, transport, layers, dtype):
    """The J-lens transport for the read `layers` on the GPU in the stream's `dtype`: None (the
    identity), one `[d, d]` map shared by every layer (an int seed, `orthogonal`, or a tensor), or
    an `[n_layers, d, d]` stack of per-layer maps (a fitted lens, `load_fitted`, or a mapping).
    A seed or a fitted spec is built once per model and kept."""
    if transport is None:
        return None
    key = (transport, tuple(layers), dtype) if isinstance(transport, (int, str)) else None
    if key in model.transports:
        return model.transports[key]
    if isinstance(transport, int):
        maps = orthogonal(model.config.hidden_size, transport)
    elif isinstance(transport, torch.Tensor):
        maps = transport
    else:
        per_layer = load_fitted(transport) if isinstance(transport, str) else transport
        maps = torch.stack([per_layer[i] for i in layers])
    maps = maps.to("cuda", dtype)
    if key is not None:
        model.transports[key] = maps
    return maps


@cell("jacobian_lens", family="llama", backend="vllm_hook")
def jacobian_lens(be, model, units, *, layers="all", position="last", transport=None):
    """The J-lens readout `[n_layers, B, vocab]` on the host: the stream after each read layer at
    each prompt's readout position (`last_token` capture for the last token, `all_tokens` for any
    other), `J_l @ h` in the stream's dtype, then the checkpoint's final RMSNorm and unembedding in
    fp32 on the client GPU (`_project`), since the plugin has no unembedding."""
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
    logits = _project(model, x.reshape(-1, x.shape[-1]), keep=True)
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
    rows = torch.cat([row for per_prompt in _token_rows(model, units, read) for row in per_prompt])
    return rows.reshape(len(read), -1, rows.shape[-1]) if layers is not None else rows


@cell("residual_readout", family="llama", backend="vllm_hook")
def residual_readout(be, model, units, *, layers):
    """The stream after each listed layer at each prompt's last token, the plugin's `last_token`
    capture: `[len(layers), B, d]` in the engine's dtype, on the host."""
    n = model.n_layers
    return _last_rows(model, units, [i % n for i in layers])


@cell("attention_pattern", family="llama", backend="vllm_hook")
def attention_pattern(be, model, units, *, layers="all", check=False):
    """The last query's attention probabilities `[n_layers, B, n_heads, k_len]`: each request lists
    the read layers in `output_qk` (attention modules numbered from 0), and the `attn_tracker`
    analyzer rebuilds every head's probabilities over the prompt from the captured post-RoPE query
    and keys (`compute_attention_from_qk`). A timed call returns them in the analyzer's dtype; a
    check call returns their log, a probability that is zero in that dtype floored first at the
    dtype's smallest normal, as in the HF cell."""
    from vllm import SamplingParams

    if model.worker != QK:
        raise _other_worker(model, QK, "Q/K capture")
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    llm = model.llm
    params = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, extra_args={"output_qk": idx})
    # MIA names a disk run by an argument; v0.2.0 keeps the last run's id on the engine object
    run_id = uuid.uuid4().hex if MIA else None
    outs = llm.generate(list(units), params, save_to_disk=model.save_to_disk, use_tqdm=False,
                        **({"run_id": run_id} if MIA and model.save_to_disk else {}))
    if model.save_to_disk:
        run_id = run_id or llm._last_run_id
        weights = llm.analyzer.compute_attention_from_qk(run_id=run_id)
        model.finished_runs.append(os.path.join(model.hook_dir, run_id))
    else:
        weights = llm.analyzer.compute_attention_from_qk(probes=outs[0].probes)
    if len(weights) != len(units):
        raise RuntimeError(f"reconstructed {len(weights)} requests' attention, expected {len(units)}")
    probs = [w[f"model.layers.{i}.self_attn.attn"]["attention"] for i in idx for w in weights]
    if check:
        probs = [torch.log(p.clamp_min(torch.finfo(p.dtype).tiny)) for p in probs]
    return torch.stack(probs).reshape(len(idx), len(units), *probs[0].shape)


# On the AsyncLLM backend (backend.py, VLLMHookAsyncBackend) the same cells read through
# `AsyncHook.capture`.
for _workload, _fn in (("logit_lens", logit_lens), ("steering", steering),
                       ("gen_steering", gen_steering), ("gen_capture", gen_capture)):
    cell(_workload, family="llama", backend="vllm_hook_async")(_fn)

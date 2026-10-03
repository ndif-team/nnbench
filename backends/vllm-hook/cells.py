"""vLLM-Hook cells for the comparison specs (design.md §12.14), built-in workers only.

Denotation, from IBM/vLLM-Hook 0e34fdd source:
- `probe_hidden_states` hooks every decoder layer and, on vLLM's fused-residual layers, records
  `output[0] + output[1]`, the stream after the layer (`probe_hidden_states_worker.hs_hook`).
  Layers are numbered from 1 ("layer N = output after the Nth block"); `last_token` keeps one row
  per forward. Capture is prefill-only by default (`hooks_on`).
- The plugin returns hidden states only, with no unembedding. The next-token readout is user code:
  the checkpoint's final RMSNorm and `lm_head` weights, applied to the last row on the GPU.
- `steer_hook_act`, the steering worker, adds a fixed vector (`add_vector`) or adjusts a projection
  (`adjust_rs`) at the LAST row of each forward of one layer: the last prompt token, and each decoded
  token when `apply_at_all_positions` is true (commit afdf36a). It is a separate worker extension,
  so an engine either reads or steers.
Workloads outside those forms are declared unsupported with the reason.
"""
import json
import os
import shutil

import torch

from isb.methodologies.registry import Unsupported, cell

ONE_ENGINE = ("an engine carries one worker extension (HookLLM worker_name), so reading the stream "
              "(probe_hidden_states) and writing it (steer_hook_act) cannot share an engine")
LAST_ROW_STEER = ("steer_hook_act writes only the last row of each forward "
                  "(steer_activation_worker._apply_steer: slice_view[-1:]), the last prompt token and "
                  "each decoded token; adding at every prompt position is not expressible")


def _weights(model):
    """The final norm and unembedding from the checkpoint, on the GPU in fp32, loaded once."""
    if not model.weights:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        root = snapshot_download(model.repo, local_files_only=True)
        index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
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


def _last_rows(model, prompt, layers):
    """Each listed layer's (0-based) last-token stream for one prompt: [len(layers), d_model]."""
    from vllm import SamplingParams

    llm = model.llm
    llm._output_layers = [i + 1 for i in layers]          # the worker numbers layers from 1
    out = llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0),
                       save_to_disk=model.save_to_disk, use_tqdm=False)
    if model.save_to_disk:
        stats = llm.analyze(analyzer_spec={"reduce": "none"})   # reads this call's run_id
        shutil.rmtree(os.path.join(llm._hook_dir, llm._last_run_id), ignore_errors=True)
    else:
        stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)
    by_layer = {name: tensors for name, tensors in stats["hidden_states"].items()}
    return torch.stack([by_layer[f"model.layers.{i}"][0].reshape(-1) for i in layers])


def _final_logits(model, prompt):
    return _project(model, _last_rows(model, prompt, [model.n_layers - 1]))   # [1, vocab]


@cell("logit_lens", family="llama", backend="vllm_hook")
def logit_lens(be, model, prompts, *, layers="all"):
    n = model.n_layers
    idx = list(range(n)) if layers == "all" else [i % n for i in layers]
    return _project(model, _last_rows(model, prompts[0], idx)).unsqueeze(1)   # [n_layers, 1, vocab]


@cell("steering", family="llama", backend="vllm_hook")
def steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, scale_by="mean_norm"):
    if alpha == 0:
        return _final_logits(model, prompts[0])
    raise Unsupported(f"{LAST_ROW_STEER}; a strength from the stream's norm also needs a read, and "
                      f"{ONE_ENGINE}")


@cell("gen_steering", family="llama", backend="vllm_hook")
def gen_steering(be, model, prompts, *, layer=8, target=" Rome", alpha=6.0, new_tokens=8,
                 scale_by="mean_norm"):
    raise Unsupported(f"{LAST_ROW_STEER}; steering at every decode step steers the last prompt "
                      f"token, not every prompt position")


@cell("activation_patching", family="llama", backend="vllm_hook")
def activation_patching(be, model, prompts, *, layer=6, patch=True, positions="all"):
    if not patch:
        return _final_logits(model, prompts[1])                # the corrupt run, unpatched
    if positions != "last":
        raise Unsupported(LAST_ROW_STEER)
    raise Unsupported("a last-token patch is an add_vector of clean minus corrupt at the last row, "
                      f"but both values must be read first, and {ONE_ENGINE}")


@cell("ablation", family="llama", backend="vllm_hook")
def ablation(be, model, prompts, *, layer=6, target="mlp"):
    if target == "none":
        return _final_logits(model, prompts[0])
    raise Unsupported("the built-in workers hook decoder layers and attention modules for reads and "
                      "steer the residual by a fixed vector; none zeroes a submodule's output")

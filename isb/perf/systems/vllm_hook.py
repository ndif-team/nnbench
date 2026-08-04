"""vllm-hook perf cell: read, steer, qk (system=vllm_hook). Needs the `vllm_hook_plugins` package.

Source-grounded against the installed plugin and run measured on GPU (bench-vllm-hook env,
vLLM 0.19.1): read, qk, and steer all execute. vllm-hook is driven
through `HookLLM`, not raw extra_args: `worker_name` selects the worker and config sets the targets.
  read  -> worker probe_hidden_states ; _output_layers + _hs_mode (last_token|all_tokens)
  qk    -> worker probe_hook_qk        ; layer_to_heads (keys = layers) + _hookq_mode
  steer -> worker steer_hook_act       ; _steering_config dict, targets a SINGLE optimal_layer and
           loads a vector .pt ({"dir": tensor}). vllm-hook steers ONE layer per request, so footprint
           half/all are not expressible here (capability boundary, not a harness bug).
"""
from __future__ import annotations

import os
import tempfile

from ..core import Config, gpu_used_mb, layer_indices, make_prompts, time_op

_WORKER = {"read": "probe_hidden_states", "qk": "probe_hook_qk", "steer": "steer_hook_act"}


def _wrap(ids):
    try:
        from vllm import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def run(cfg: Config) -> dict:
    import torch
    from vllm import SamplingParams
    from vllm_hook_plugins import HookLLM

    worker = _WORKER.get(cfg.op)
    if worker is None:
        raise NotImplementedError(f"vllm-hook has no op {cfg.op!r}")

    mem0 = gpu_used_mb()
    hook_dir = tempfile.mkdtemp(prefix="vllm_hook_")
    # download_dir=None -> vLLM resolves the model via HF_HOME (the cached models), not HookLLM's
    # default ~/.cache; hook_dir is passed explicitly so it does not depend on download_dir.
    llm = HookLLM(
        model=cfg.repo, worker_name=worker, download_dir=None, hook_dir=hook_dir,
        enforce_eager=cfg.enforce_eager, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    hf = llm.llm_engine.model_config.hf_config
    layers = layer_indices(cfg, hf.num_hidden_layers)

    if cfg.op == "read":
        llm._output_layers = list(layers)
        llm._hs_mode = cfg.token_mode
    elif cfg.op == "qk":
        llm.layer_to_heads = {int(L): [0] for L in layers}   # keys = layers; heads only matter to the analyzer
        llm._hookq_mode = cfg.token_mode
    elif cfg.op == "steer":
        if cfg.footprint != "one":
            raise NotImplementedError("vllm-hook steers a single optimal_layer per request")
        vec_path = os.path.join(hook_dir, "steer.pt")
        torch.save({"dir": torch.zeros(hf.hidden_size, dtype=torch.float32)}, vec_path)
        llm._steering_config = {
            "method": "add_vector", "coefficient": 1.0, "optimal_layer": int(layers[0]),
            "vector_path": vec_path, "apply_at_all_positions": True,
        }

    prompts = [_wrap(ids) for ids in make_prompts(cfg)]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.eff_new_tokens())

    def once():
        llm.generate(prompts, sp)              # HookLLM injects the worker's extra_args internally
        return None, None

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps, mem0=mem0)
    metrics.update({"artifact_kb": 0.0, "transfer_bytes": 0, "correct": None})
    return metrics

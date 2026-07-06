"""vllm-lens perf cell: read, steer (system=vllm_lens). Needs the `vllm-lens` plugin installed.

SCAFFOLD. The read path matches their own benchmark (extra_args output_residual_stream); the steer
path uses their steering-vector config and is marked VERIFY (exact key/object). No Q/K path (they
do not expose one). Plugin auto-registers on install.
"""
from __future__ import annotations

from ..core import Config, gpu_used_mb, layer_indices, make_prompts, time_op


def _wrap(ids):
    try:
        from vllm import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def run(cfg: Config) -> dict:
    import vllm_lens  # noqa: F401  (import-for-side-effect: registers the plugin)
    from vllm import LLM, SamplingParams

    if cfg.op == "qk":
        raise NotImplementedError("vllm-lens has no Q/K path")

    mem0 = gpu_used_mb()
    llm = LLM(
        model=cfg.repo, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
    )
    prompts = [_wrap(ids) for ids in make_prompts(cfg)]
    n_new = cfg.eff_new_tokens()
    hf = llm.llm_engine.model_config.hf_config
    nlayers = hf.num_hidden_layers
    hidden = hf.hidden_size
    layers = layer_indices(cfg, nlayers)

    extra = {}
    if cfg.op == "read":
        extra["output_residual_stream"] = layers          # source-confirmed (_worker_ext.py:257)
    elif cfg.op == "steer":
        # source-confirmed (types.py / activation_oracle.py): apply_steering_vectors takes
        # SteeringVector objects. One SV with activations [n_layers, hidden] + layer_indices covers
        # the footprint (multi-layer steer is supported here). Value is perf-only.
        import torch
        from vllm_lens import SteeringVector
        sv = SteeringVector(
            activations=torch.zeros(len(layers), hidden, dtype=torch.float32),
            layer_indices=list(layers), scale=1.0,
        )
        extra["apply_steering_vectors"] = [sv]
    sp = SamplingParams(temperature=0.0, max_tokens=n_new, extra_args=extra)

    def once():
        llm.generate(prompts, sp)              # activations attach to the RequestOutput
        return None, None

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps, mem0=mem0)
    metrics.update({"artifact_kb": 0.0, "transfer_bytes": 0, "correct": None})
    return metrics

"""Native vLLM read baseline: the in-tree ExampleHiddenStatesConnector (system=native_eagle).

vLLM's own hidden-state extraction path: the `extract_hidden_states` speculative method inserts one
CacheOnlyAttentionLayer that caches the selected layers' hidden states (Eagle-3 aux-hidden-state
infra, hence the drafter memory overhead captured by peak_mem), and the connector writes them to a
safetensors file per request. Wiring matches vLLM v0.19.1
examples/offline_inference/extract_hidden_states.py. Layer selection comes from
`eagle_aux_hidden_state_layer_ids`, so the footprint axis maps directly. Read only; no steer, no Q/K.
Extraction covers prompt tokens (the connector saves at prefill). Multi-token generation crashes
the engine on vLLM 0.19.1: the drafter schedules a placeholder spec token (-1) that reaches an
embedding gather and trips a device-side assert after a few decode steps (upstream ships this path
with max_tokens=1 only). Decode cells therefore record an engine-crash error row; that is the
measured capability boundary of the native path, kept as a result.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ..core import Config, gpu_used_mb, layer_indices, make_prompts, time_op


def _wrap(ids):
    try:
        from vllm import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def run(cfg: Config) -> dict:
    if cfg.op not in ("read", "none"):
        raise NotImplementedError("native Eagle connector is read-only")
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    layers = layer_indices(cfg, AutoConfig.from_pretrained(cfg.repo).num_hidden_layers)
    store = tempfile.mkdtemp(prefix="eagle_hs_")
    mem0 = gpu_used_mb()
    llm = LLM(
        model=cfg.repo, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [int(L) for L in layers]}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": store},
        },
    )
    prompts = [_wrap(ids) for ids in make_prompts(cfg)]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.eff_new_tokens())

    def once():
        llm.generate(prompts, sp)              # hidden states land as safetensors files in `store`
        return None, None

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps, mem0=mem0)
    files = sorted(Path(store).glob("*.safetensors"))
    metrics.update({
        "artifact_kb": round(files[-1].stat().st_size / 1024, 1) if files else 0.0,
        "transfer_bytes": 0, "correct": None,
    })
    shutil.rmtree(store, ignore_errors=True)
    return metrics

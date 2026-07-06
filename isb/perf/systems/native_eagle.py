"""Native vLLM read baseline: the in-tree ExampleHiddenStatesConnector (system=native_eagle).

SCAFFOLD. This is vLLM's own hidden-state extraction path (built on Eagle-3 spec-decode infra, which
is why it carries a drafter memory overhead, captured by peak_mem). Read only; no steer, no Q/K. The
connector wiring (kv_transfer_config / connector args) must be VERIFIED against the installed vLLM:
see docs.vllm.ai .../examples/offline_inference/extract_hidden_states/.
"""
from __future__ import annotations

from ..core import Config, make_prompts, time_op


def _wrap(ids):
    try:
        from vllm import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}


def run(cfg: Config) -> dict:
    if cfg.op not in ("read", "none"):
        raise NotImplementedError("native Eagle connector is read-only")
    from vllm import LLM, SamplingParams

    # VERIFY: the exact connector config for the installed vLLM. The connector extracts all-layer
    # hidden states for the prompt; it has no per-layer or last-token selection (note in results).
    llm = LLM(
        model=cfg.repo, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
        # kv_transfer_config=...  # ExampleHiddenStatesConnector — fill per the vLLM example
    )
    prompts = [_wrap(ids) for ids in make_prompts(cfg)]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.eff_new_tokens())

    def once():
        llm.generate(prompts, sp)
        return None, None

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps)
    metrics.update({"artifact_kb": 0.0, "transfer_bytes": 0, "correct": None})
    return metrics

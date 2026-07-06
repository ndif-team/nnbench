"""Baseline: vanilla vLLM generation, no intervention. The overhead denominator (system=pure_vllm).

Written for the nnsight-serve-test env. Not run in CI (needs a GPU). gen == total here (no retrieval).
"""
from __future__ import annotations

from ..core import Config, gpu_used_mb, make_prompts, time_op


def _wrap(ids):
    try:
        from vllm import TokensPrompt
        return TokensPrompt(prompt_token_ids=ids)
    except Exception:
        return {"prompt_token_ids": ids}      # older vLLM accepts the dict form


def run(cfg: Config) -> dict:
    from vllm import LLM, SamplingParams

    mem0 = gpu_used_mb()                        # NVML baseline before model load (footprint delta)
    llm = LLM(
        model=cfg.repo, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=cfg.enforce_eager,
    )
    prompts = [_wrap(ids) for ids in make_prompts(cfg)]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.eff_new_tokens())

    def once():
        llm.generate(prompts, sp)
        return None, None                      # no split: time_op's total IS the gen time here

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps, mem0=mem0)
    metrics.update({"artifact_kb": 0.0, "transfer_bytes": 0, "correct": None})
    return metrics

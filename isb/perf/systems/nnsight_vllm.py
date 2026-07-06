"""nnsight-on-vLLM perf cell: read / steer / qk (system=nnsight_vllm).

SCAFFOLD. The structure (engine config, prompts, op over resolved blocks, split timer) is final,
but the exact trace idioms (decode iteration, multi-prompt invokes, the steer write form, the Q/K
access path) must be VERIFIED in the nnsight-serve-test env against the current nnsight vLLM
contract. Read forms follow isb/methodologies/logit_lens.py; steer follows steering.py. Marked
inline where verification is required.
"""
from __future__ import annotations

import torch

from ..core import Config, gpu_used_mb, layer_indices, make_prompts, time_op


def _blocks(model):
    """Decoder block list. Structural probe over the two known conventions; raises loudly if neither
    (no silent default). The full benchmark should use nnbench family profiles instead of this lazy
    resolver; it is adequate for the fixed model ladder the perf micro runs on."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers                      # Llama/Qwen/OPT style
    tr = getattr(model, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return tr.h                              # GPT-2 style
    raise AttributeError(f"cannot locate decoder blocks on {type(model).__name__}")


def _resid(out):
    # untuple: real tuples and LazyRemoteTensor both index [0]; bare tensors pass through.
    return out if isinstance(out, torch.Tensor) else out[0]


def run(cfg: Config) -> dict:
    from nnsight.modeling.vllm import VLLM

    mem0 = gpu_used_mb()                         # NVML baseline before model load (footprint delta)

    # nnsight's VLLM forces eager itself and forwards it to vLLM, so passing enforce_eager here
    # duplicates the LLM() kwarg. The integration is eager by construction (matches cfg.enforce_eager
    # for the baseline); the existing isb async backend likewise never passes it.
    kw = dict(
        dispatch=True, dtype=cfg.dtype, max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    if cfg.tensor_parallel_size > 1:
        kw["tensor_parallel_size"] = cfg.tensor_parallel_size
    model = VLLM(cfg.repo, **kw)                 # sync mode; async/serve are separate systems later

    ids = make_prompts(cfg)
    prompt = ids[0]                              # VERIFY: multi-prompt = per-prompt tracer.invoke (batch>1 TODO)
    blocks = _blocks(model)
    idx = layer_indices(cfg, len(blocks))
    n_new = cfg.eff_new_tokens()

    # steer direction: a fixed unit vector at the block's hidden size (perf only; value irrelevant).
    steer_vec = None

    def _do_op():
        nonlocal steer_vec
        if cfg.op == "read":
            return [_resid(blocks[i].output).save() for i in idx]
        if cfg.op == "qk":
            # Q/K capture = read the attention qkv-projection output (the merged QKV carries Q, K, V;
            # comparable to vllm-hook's Q/K capture). The attention PATTERN is unreadable on vLLM
            # (paged/flash never materializes it; attention_pattern.py confirms it ERRORs), so this
            # reads projections, not weights. Structural probe over the family's attn + qkv names.
            saved = []
            for i in idx:
                attn = getattr(blocks[i], "self_attn", None) or getattr(blocks[i], "attn", None)
                if attn is None:
                    raise AttributeError(f"no attention submodule on block {i}")
                qkv = (getattr(attn, "qkv_proj", None) or getattr(attn, "c_attn", None)
                       or getattr(attn, "q_proj", None))
                if qkv is None:
                    raise AttributeError(f"no qkv projection on {type(attn).__name__}")
                saved.append(_resid(qkv.output).save())
            return saved
        if cfg.op == "steer":
            # Whole-tuple replacement is the vLLM-safe write (steering.py: in-place `[:] =` hits
            # InferenceMode protection and ERRORs). Value is irrelevant (perf only).
            with torch.no_grad():
                for i in idx:
                    out = blocks[i].output
                    is_tuple = isinstance(out, tuple)
                    hidden = out[0] if is_tuple else out
                    if steer_vec is None or steer_vec.shape[-1] != hidden.shape[-1]:
                        steer_vec = torch.zeros(hidden.shape[-1], dtype=hidden.dtype,
                                                device=hidden.device) + 1e-3
                    new_hidden = hidden + steer_vec
                    blocks[i].output = (new_hidden, *out[1:]) if is_tuple else new_hidden
            return None
        return None

    def once():
        if cfg.phase == "prefill":
            with model.trace(prompt, temperature=0.0, max_tokens=1):
                _do_op()
            return None, None
        # decode: per-step op under bounded iteration (the vLLM-safe realization, isb backend §generate)
        with model.trace(prompt, temperature=0.0, max_tokens=n_new) as tracer:
            for _ in tracer.iter[0:n_new]:      # VERIFY idiom against current nnsight
                _do_op()
        return None, None

    metrics, _ = time_op(once, n_warmup=cfg.n_warmup, n_reps=cfg.n_reps, mem0=mem0)
    metrics.update({"artifact_kb": 0.0, "transfer_bytes": 0, "correct": None})
    return metrics

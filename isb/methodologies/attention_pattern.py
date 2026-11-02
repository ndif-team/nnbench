"""Attention-pattern read: the post-softmax attention probabilities of the last query token, as
log-probabilities so the oracle's softmax recovers them, `[n_layers, B, n_heads, k_len]`. HF eager
attention materializes them (GPT-2: `attn.source.attention_interface_1.output[1]`; Llama-tree
models: `self_attn.output[1]`); sdpa and vLLM's attention kernels never do. On vLLM the gpt2 read
raises (the recorded verdict); the llama cell reads the kernel's query and key inputs and computes
the last query's softmax beside the kernel."""
from __future__ import annotations

import torch

from .registry import Unsupported, cell


def _attn_pattern(blocks, *, layers):
    """Stack the last-query attention pattern (log-probs) over the requested layers. Runs inside a
    trace."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():
        for i in idx:
            weights = blocks[i].attn.source.attention_interface_1.output[1]   # [B, heads, q, k]
            rows.append(torch.log(weights[:, :, -1, :]))                      # [B, heads, k]
    return torch.stack(rows, dim=0)


@cell("attention_pattern", family="gpt2", backend="hf")
def attention_pattern_gpt2_hf(be, model, units, *, layers="all"):
    def build(rows):
        return _attn_pattern(model.transformer.h, layers=layers)

    return be.run(model, units, build)


@cell("attention_pattern", family="gpt2", backend="vllm_async")
def attention_pattern_gpt2_vllm(be, model, units, *, layers="all"):
    def build(rows):
        return _attn_pattern(model.transformer.h, layers=layers)

    return be.run(model, units, build)


def _eager_pattern(blocks, *, layers):
    """HF Llama-tree attention (transformers 5.12.1 `Qwen2Attention.forward`, the same return on
    Llama and Qwen3) returns `(attn_output, attn_weights)`; under eager, `attn_weights` is
    `eager_attention_forward`'s softmax over `QK^T * scaling + mask`, `[B, heads, q, k]`, the KV
    heads already repeated to the query heads. The documented nnsight read is `self_attn.output[1]`
    (nnsight docs/patterns/attention-patterns.md). A weight that underflowed to zero in the cast to
    the model dtype (or a left-pad key) is floored at the dtype's smallest normal before the log, so
    the output stays finite. Runs inside a trace."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():
        for i in idx:
            weights = blocks[i].self_attn.output[1][:, :, -1, :]      # [B, heads, k]
            rows.append(torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny)))
    return torch.stack(rows, dim=0)


@cell("attention_pattern", family="llama", backend="hf")
def attention_pattern_llama_hf(be, model, units, *, layers="all"):
    if be.attn_implementation != "eager":
        raise Unsupported(
            f"attn_implementation={be.attn_implementation!r} returns no attention weights: "
            "transformers 5.12.1 sdpa_attention_forward returns (attn_output, None) "
            "(integrations/sdpa_attention.py:104); only eager materializes them")

    def build(rows):
        return _eager_pattern(model.model.layers, layers=layers)

    return be.run(model, units, build)


def _kernel_input_pattern(blocks, *, layers):
    """The last query's attention log-probabilities from the inputs of vLLM's attention kernel.

    Denotation (vLLM 0.19.1 and 0.27.1, `models/qwen2.py` `Qwen2Attention.forward`; `llama.py` and
    `qwen3.py` the same order): `qkv_proj` is one fused projection split into q, k, v; Qwen3's
    q/k norms run next; `rotary_emb` then rotates q and k in place (`layers/rotary_embedding/
    base.py:221`, "an in-place operation"); `self.attn(q, k, v)` is the kernel. So the kernel module's
    positional inputs are the post-RoPE query `[tokens, n_heads * head_size]` and key
    `[tokens, n_kv_heads * head_size]` (`layers/attention/attention.py`, `Attention.forward`), and
    its softmax scale is the attention module's `scaling` (passed as the kernel's `scale`). Read
    there, the engine's own RoPE is in the values; nothing is recomputed.

    The served rows are this request's prefill tokens: nnsight refuses a prompt split by chunked
    prefill (`GPUModelRunner.Requests.refuse_chunked`) and every nnsight-vllm config turns prefix
    caching off, so the keys are the whole prompt and the last query attends to all of them (full
    causal attention; the check below refuses a sliding window). Query head `h` reads KV head
    `h // (n_heads // n_kv_heads)`, as HF's `repeat_kv` and the kernels do. The scores are taken in
    fp32, the precision of the kernel's softmax. Runs inside a trace, in the worker."""
    idx = range(len(blocks)) if layers == "all" else layers
    rows = []
    with torch.no_grad():
        for i in idx:
            attn = blocks[i].self_attn
            kernel = attn.attn
            if kernel.sliding_window is not None:
                raise ValueError(f"layer {i}: sliding-window attention; the last query does not "
                                 "attend to every prompt key")
            args, _ = kernel.inputs
            q, k = args[0], args[1]
            size = kernel.head_size
            n_heads, n_kv = q.shape[-1] // size, k.shape[-1] // size
            query = q[-1].reshape(n_kv, n_heads // n_kv, size).float()      # [kv, group, d]
            keys = k.reshape(k.shape[0], n_kv, size).float()                 # [tokens, kv, d]
            scores = torch.einsum("grd,sgd->grs", query, keys).reshape(n_heads, -1) * attn.scaling
            rows.append(torch.log_softmax(scores, dim=-1))                   # [heads, k]
    return torch.stack(rows, dim=0).unsqueeze(1)                             # [n_layers, 1, heads, k]


@cell("attention_pattern", family="llama", backend="vllm_async")
def attention_pattern_llama_vllm(be, model, units, *, layers="all"):
    if len(units) != 1:
        # k_len is each prompt's own length, and the backends stack units along dim -2 (the heads
        # axis here), so several units have no shared layout
        raise ValueError(f"attention_pattern on vLLM takes one unit per call, got {len(units)}")
    blocks = model.model.layers        # bound outside the block: the block ships no root model

    def build(rows):
        return _kernel_input_pattern(blocks, layers=layers)

    return be.run(model, units, build)

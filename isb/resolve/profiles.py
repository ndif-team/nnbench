"""Family + Backend profiles (design.md §11.4, §11.6) — DATA, the only place
concrete module paths live. Adding a family = a new FamilyProfile entry, never code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Capability tokens = logical targets (spec.TARGETS) PLUS access/feature tokens.
# `predict()` intersects family.caps & backend.caps against a workload's needs.
_ALL_TARGETS = {
    "block.output", "block.input", "attn.output", "mlp.output",
    "attn.head_value", "mlp.neuron", "attn.weights", "logits", "sampled_token",
}
_ALL_ACCESS = {"read", "write_replace", "write_inplace", "cache", "grad"}
# Motif primitives that are universally available wherever reads are (so they do NOT
# gate at predict-time). Gating primitives (grad, source, attn.weights) live in caps
# and CAN be absent on a backend. "aux" = run a side module/computation on a value.
_PRIMITIVES = {"cache", "aux"}


@dataclass
class BackendProfile:
    """Per-backend realizable capability set. §11.6."""

    name: str
    caps: set


@dataclass
class FamilyProfile:
    """Per (model-type, family) binding map. §11.4."""

    type: str                  # causal_lm | diffusion | vlm
    family: str
    paths: dict                # role -> path template ("{i}" = layer index)
    output_index: dict         # role -> int|None  (tuple-output handling hint)
    dims: dict                 # logical dim -> config attribute name (None => derive)
    caps: set                  # capability tokens this architecture exposes
    n_layers_attr: str         # config attribute giving #blocks

    def layer_path_role(self, target: str) -> str:
        """Which path-template role realizes a logical target."""
        role = {
            "block.output": "block",
            "block.input": "block",
            "attn.output": "attn",
            "mlp.output": "mlp",
            "attn.head_value": "attn_oproj",  # per-head value = o_proj/c_proj INPUT (§11.3)
            "mlp.neuron": "mlp_neuron",       # alias resolved below
        }[target]
        if role == "mlp_neuron":
            # families name the neuron-bearing module differently
            for k in ("mlp_act", "mlp_down"):
                if k in self.paths:
                    return k
            raise KeyError("no mlp_act/mlp_down path for mlp.neuron")
        return role


# Architectural caps shared by standard decoder-only causal LMs.
_CAUSAL_LM_CAPS = _ALL_TARGETS | _ALL_ACCESS | _PRIMITIVES

GPT2 = FamilyProfile(
    type="causal_lm",
    family="gpt2",
    paths={
        "block": "transformer.h.{i}",
        "attn": "transformer.h.{i}.attn",
        "attn_oproj": "transformer.h.{i}.attn.c_proj",
        "mlp": "transformer.h.{i}.mlp",
        "mlp_act": "transformer.h.{i}.mlp.act",
        "final_norm": "transformer.ln_f",
        "unembed": "lm_head",
    },
    output_index={"block": 0, "attn": 0, "mlp": None},
    dims={
        "n_heads": "n_head", "head_dim": None, "n_kv_heads": "n_head",
        "ffn": "n_inner", "hidden": "n_embd",   # n_inner may be None -> derive 4*hidden
    },
    caps=set(_CAUSAL_LM_CAPS),
    n_layers_attr="n_layer",
)

LLAMA = FamilyProfile(
    type="causal_lm",
    family="llama",
    paths={
        "block": "model.layers.{i}",
        "attn": "model.layers.{i}.self_attn",
        "attn_oproj": "model.layers.{i}.self_attn.o_proj",
        "mlp": "model.layers.{i}.mlp",
        "mlp_down": "model.layers.{i}.mlp.down_proj",
        "final_norm": "model.norm",
        "unembed": "lm_head",
    },
    output_index={"block": 0, "attn": 0, "mlp": None},
    dims={
        "n_heads": "num_attention_heads",
        "head_dim": "head_dim",
        "n_kv_heads": "num_key_value_heads",  # GQA-aware
        "ffn": "intermediate_size",
        "hidden": "hidden_size",
    },
    caps=set(_CAUSAL_LM_CAPS),
    n_layers_attr="num_hidden_layers",
)

# Map HF config.model_type -> profile. Extend here, never in the resolver.
FAMILY_REGISTRY = {
    "gpt2": GPT2,
    "llama": LLAMA,
    "mistral": LLAMA,   # same block/attn/mlp layout
    "qwen2": LLAMA,
    "qwen3": LLAMA,
}


def family_for(model_type: str) -> FamilyProfile:
    """Resolve a FamilyProfile from a HF `config.model_type` string."""
    key = (model_type or "").lower()
    if key not in FAMILY_REGISTRY:
        raise KeyError(
            f"no FamilyProfile for model_type {model_type!r}; "
            f"add one to FAMILY_REGISTRY (known: {sorted(FAMILY_REGISTRY)})"
        )
    return FAMILY_REGISTRY[key]


# --- Backend profiles (§11.6) ---------------------------------------------------

HF = BackendProfile(
    name="hf",
    # eager HF realizes everything, incl. attn.weights + grad + source
    caps=_ALL_TARGETS | _ALL_ACCESS | _PRIMITIVES | {"source"},
)

VLLM_ASYNC = BackendProfile(
    name="vllm_async",
    caps={
        "block.output", "block.input", "attn.output", "mlp.output",
        "attn.head_value", "mlp.neuron", "logits", "sampled_token",
        "read", "write_replace", "write_inplace", "cache", "aux",
    },  # NO attn.weights (flash-attn), NO grad (inference engine), NO source
)

BACKEND_REGISTRY = {HF.name: HF, VLLM_ASYNC.name: VLLM_ASYNC}

"""Model profiles: the model-implementation index (design.md §12.8).

A profile is DATA: where a family's blocks / final norm / unembed head live in the module
tree, plus the family's residual denotation on vLLM (fused `(hidden, residual)` tuples vs a
plain tensor). This is the implementation half of "model knowledge"; the identity half
(repo, load kwargs) stays in specs. Before this module the same module paths were
copy-pasted into every methodology file (~55 hardcoded occurrences), so adding a family
meant editing ~8 files; now it is one row here.

Family-generic cells register `family="*"` and receive the profile as `m` (bound by
`registry.get_cell`). An explicit `(methodology, family, backend)` registration still takes
precedence: the override hatch for a family whose intervention code genuinely differs,
which is exactly the case worth reading explicitly. The profile only says WHERE modules
are, never WHAT the intervention does; nothing here constructs intervention code (the §11
Resolver lesson still holds).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


def _untuple(x):
    # PP cross-stage outputs are LazyRemoteTensor regardless of what they wrap, so isinstance
    # answers "is this a lazy", never "is the wrapped value a tensor or a tuple". Indexing [0] on
    # every lazy (the 0.7-era convention this helper used to follow) is correct for tuple-output
    # blocks (Qwen/Llama: element 0 is the hidden state) and wrong for tensor-output blocks
    # (GPT-2: it selects ROW 0 of the (tokens, hidden) slab, a 1-D tensor that breaks every
    # consumer downstream). Ask the lazy for the wrapped value's rank instead: `.ndim`
    # materializes a wrapped tensor and answers; for a wrapped tuple the lazy deliberately
    # raises its index-it-first AttributeError, which is the designed discriminator, not a hedge.
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, tuple):
        return x[0]
    try:
        x.ndim
        return x
    except AttributeError:
        return x[0]


def _resid(out, how):
    """The residual stream at a block's output boundary.

    `plain`: the block output (or `[0]` of its tuple), correct for GPT-2 and for any HF block,
    which already carries the full residual stream.
    `fused`: `hidden + residual`. This is the **documented vLLM dual-residual-stream issue** (the
    vLLM dual residual stream): vLLM's Llama/RMSNorm decoder layers return `(hidden, residual)` and
    the true residual stream is their SUM; vLLM computes exactly `hidden + residual` for its own aux
    hidden states (vllm .../models/llama.py:425; VLLM_GUIDE "Logit Lens" prescribes `out[0] + out[1]`).
    Reading only `[0]` (as `plain` does) drops the accumulated residual -> silently wrong logits.
    Falls back to `plain` when the output is not a `(hidden, residual)` container, so it is safe on HF.
    The second branch handles **pipeline parallelism**: a PP cross-stage output is a `LazyRemoteTensor`
    wrapping `(hidden, residual)`, NOT a tuple instance, so `isinstance(out, tuple)` alone would skip
    the fused branch and silently drop the residual on the PP candidate (the same reason `_untuple`
    probes tensor-ness). `fused` is only ever requested by vLLM dual-residual cells, where a non-tensor
    output always carries `(hidden, residual)`.
    """
    if how == "fused":
        if isinstance(out, tuple) and len(out) >= 2:        # real tuple (HF / single-GPU vLLM)
            return out[0] + out[1]
        if not isinstance(out, (torch.Tensor, tuple)):      # PP LazyRemoteTensor wrapping (hidden, residual)
            return out[0] + out[1]
    return _untuple(out)


def _attr(obj, path: str):
    for name in path.split("."):
        obj = getattr(obj, name)                            # a wrong path raises loudly, by design
    return obj


@dataclass(frozen=True)
class ModelProfile:
    """One architecture's module locations. Paths are dotted attribute paths from the model root;
    they are the same on HF and on nnsight's vLLM wrapper (both expose the HF module tree).

    Scope: the profile carries BOUNDARY-tier locations only (block list, final norm, head, and the
    within-block attn/mlp submodule names). Internal-tier `.source` op names stay out: they are
    family x implementation specific (e.g. GPT-2 HF-eager's `attention_interface_0`) and belong to
    the explicit cells that measure them (attention_pattern stays per-family for exactly this
    reason)."""
    family: str
    blocks_path: str            # the decoder block list
    norm_path: str              # the final norm before the unembed
    head_path: str = "lm_head"
    # A block-output's DENOTATION is per (family x engine), not per family alone (design §3.2: the
    # same site name means different things per context): HF blocks carry the full residual stream
    # in [0] ("plain") for every family, while vLLM's implementation of RMSNorm families returns
    # (hidden, residual) whose sum is the stream ("fused"). It cannot live on the backend (backends
    # would each need a family table) and cannot be runtime-detected (HF llama blocks ALSO return
    # tuples, whose [1] is attention metadata, not a residual — shape probing cannot tell them
    # apart), so the family row carries it keyed by engine kind. An engine kind with no entry fails
    # loudly: an undeclared denotation read is exactly the SILENTLY_WRONG shape.
    residual_denotation: dict = field(default_factory=lambda: {"hf": "plain", "vllm": "plain"})
    # within-block submodule names (ablation targets); None = the family has no such split (e.g.
    # nemotron's single per-layer `mixer`, whose ablation cells stay explicit)
    attn_name: str | None = "attn"
    mlp_name: str | None = "mlp"
    # Some architectures mount the SAME text tree under an extra prefix on vLLM: Qwen3.5's
    # multimodal wrapper is `model.layers` on HF but `language_model.model.layers` on the nnsight
    # vLLM envoy (measured 2026-07-22 on Qwen/Qwen3.5-4B: `model.layers` absent, prefixed tree
    # present). The registry binds the engine view via `for_backend`, so cells never see this.
    vllm_prefix: str = ""

    def for_backend(self, backend_name: str) -> "ModelProfile":
        """The engine-specific view of this profile: identity on HF or when no prefix is declared;
        on vLLM, the same profile with every module path behind the wrapper prefix."""
        if not self.vllm_prefix or not backend_name.startswith("vllm"):
            return self
        p = self.vllm_prefix
        return replace(self, blocks_path=f"{p}.{self.blocks_path}",
                       norm_path=f"{p}.{self.norm_path}", head_path=f"{p}.{self.head_path}",
                       vllm_prefix="")

    def default_residual(self, backend_name: str) -> str:
        kind = "vllm" if backend_name.startswith("vllm") else backend_name
        return self.residual_denotation[kind]       # unknown engine kind raises loudly, by design

    def blocks(self, model):
        return _attr(model, self.blocks_path)

    def norm(self, model):
        return _attr(model, self.norm_path)

    def head(self, model):
        return _attr(model, self.head_path)

    def submodule(self, block, which: str):
        """The within-block ablation target ("attn" | "mlp") by this family's own name."""
        name = {"attn": self.attn_name, "mlp": self.mlp_name}[which]
        if name is None:
            raise ValueError(
                f"family {self.family!r} has no within-block {which!r} submodule (single-op blocks); "
                f"use that family's explicit cells"
            )
        return getattr(block, name)

    # the documented read pattern lives with the profile so tasks/cells share one copy
    resid = staticmethod(_resid)


_FUSED_ON_VLLM = {"hf": "plain", "vllm": "fused"}

PROFILES = {
    "gpt2": ModelProfile("gpt2", "transformer.h", "transformer.ln_f"),
    # llama covers the Llama-tree families the specs already bind to it (SmolLM2, Qwen2.5: same
    # model.layers/model.norm/lm_head tree; specs/qwen.py documents the reuse)
    "llama": ModelProfile("llama", "model.layers", "model.norm",
                          residual_denotation=_FUSED_ON_VLLM, attn_name="self_attn"),
    # NemotronH hybrid (Mamba/attention/MoE blocks): same tree on HF and vLLM, norm is norm_f (§12.7);
    # blocks are single-op (`mixer`), so there is no attn/mlp split (ablation cells stay explicit)
    "nemotron": ModelProfile("nemotron", "model.layers", "model.norm_f",
                             residual_denotation=_FUSED_ON_VLLM, attn_name=None, mlp_name=None),
    # Qwen3.5 (hybrid linear-attention, e.g. Qwen3.5-4B): HF auto-routes the text-only repo to
    # Qwen3_5ForCausalLM (llama-shaped tree), while vLLM builds the multimodal wrapper, so the same
    # tree sits behind `language_model.` there (the vllm_prefix). vLLM decoder layers return
    # (hidden, residual) like the other RMSNorm families. Blocks mix `linear_attn` and `self_attn`
    # per layer_types, so there is no uniform attn submodule name.
    "qwen3_5": ModelProfile("qwen3_5", "model.layers", "model.norm",
                            residual_denotation=_FUSED_ON_VLLM, attn_name=None,
                            vllm_prefix="language_model"),
}

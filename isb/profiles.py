"""Model profiles: the model-implementation index (design.md §12.8).

A profile is DATA: where a family's blocks / final norm / unembed head live in the module
tree, plus the family's residual denotation on vLLM (fused `(hidden, residual)` tuples vs a
plain tensor). The rows live torch-free in `isb/models.py` (`FAMILIES`, §12.16) beside the
checkpoints; this module adds the torch helpers and module accessors the cells call. Before
profiles the same module paths were copy-pasted into every methodology file (~55 hardcoded
occurrences), so adding a family meant editing ~8 files; now it is one row.

Family-generic cells register `family="*"` and receive the profile as `m` (bound by
`registry.get_cell`). An explicit `(methodology, family, backend)` registration still takes
precedence: the override hatch for a family whose intervention code genuinely differs,
which is exactly the case worth reading explicitly. The profile only says WHERE modules
are, never WHAT the intervention does; nothing here constructs intervention code (the §11
Resolver lesson still holds).
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from .models import FAMILIES, Family


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
class ModelProfile(Family):
    """A family row (isb/models.py) with the module accessors the cells call."""

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


PROFILES = {name: ModelProfile(**{f.name: getattr(row, f.name) for f in fields(Family)})
            for name, row in FAMILIES.items()}

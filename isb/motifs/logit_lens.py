"""Logit lens motif (design.md §11.7 worked example).

Family-independent: reads `block.output` at every layer (resolved Bindings), applies
the model's own final norm + unembed, and stacks per-layer logits. It NEVER references
a concrete module path — `read_value` and the resolved `norm`/`unembed` envoys carry
all model knowledge. Backend-shape differences (HF [B,S,H] vs vLLM flat [tokens,H]) are
handled by the injected `ctx.select_last` / `ctx.stack`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..resolve import read_value
from .registry import motif


@dataclass
class Program:
    """A backend-agnostic trace program: build the single proxy tensor to save."""

    build_proxy: Callable[[Any, Any], Any]  # (model, ctx) -> proxy tensor (un-saved)
    site_ids: list
    requires: set


@motif("logit_lens", requires={"cache", "aux"})
def logit_lens(workload, resolver):
    sel = workload.selectors[0]
    sites = resolver.resolve(sel)                 # block.output across scope
    norm = resolver.resolve_one("final_norm").module
    unembed = resolver.resolve_one("unembed").module
    site_ids = [b.site_id for b in sites]
    # unembed via the module's forward ("module", idiomatic) or a weight matmul
    # ("weight", portable). vLLM's ParallelLMHead.forward() guards against direct
    # calls ("LMHead's weights should be used in the sampler"), so "weight" is the
    # portable path; "module" stays as the canonical-form frontier marker.
    unembed_mode = workload.params.get("unembed", "module")

    def build_proxy(model, ctx):
        import torch

        rows = []
        # Logit lens is forward-only. no_grad keeps the aux head (grad-enabled params)
        # from tracking vLLM's inference-mode intermediates for backward — required on
        # vLLM, harmless on HF (one motif, both backends).
        with torch.no_grad():
            for b in sites:
                hs = read_value(b)                    # residual stream at this layer
                normed = norm(hs)
                if unembed_mode == "weight":
                    logits = torch.nn.functional.linear(normed, unembed.weight)
                else:
                    logits = unembed(normed)          # model's own head (forward dispatch)
                rows.append(ctx.select_last(logits))  # [1, vocab]
            return ctx.stack(rows)                    # [n_layers, 1, vocab]

    return Program(build_proxy, site_ids, {"cache", "aux"})

"""CausaLab-aligned semantic indexing for nnbench cells.

This module deliberately describes interventions; it does not construct them.  The executable
program remains the explicit ``@cell(methodology, family, backend)`` function.  Keeping this layer
non-executable preserves nnbench's central invariant: backend realizations are things the benchmark
measures, not generated implementation details hidden behind a resolver.

``InterventionSpec`` is the intersection of CausaLab's intervention-protocol vocabulary and the
coordinates nnbench needs to index a cell.  It is intentionally smaller than a runnable CausaLab
document: model and dataset bindings come from ``CellConfig``/``ExecutionRegime``; metrics, saves,
artifacts, and workflows remain CausaLab concerns.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


COMPONENTS = frozenset({
    "embeddings", "block_input", "block_output", "attention_output", "attention_value",
    "attention_probs", "mlp_input", "mlp_output", "mlp_activation", "router_logits",
    "expert_output", "ln_final", "lm_head",
})
OPERATIONS = frozenset({"read", "write", "grad"})
MECHANISMS = frozenset({
    "swap", "add_scaled", "lerp", "affine", "gaussian", "renormalize", "clamp",
    "pytorch_fn",
})
POSITION_FRAMES = frozenset({"prompt", "generated"})
FEATURIZERS = frozenset({"subspace", "gate"})
CAPABILITIES = frozenset({
    "grad", "paired_forward", "full_logits", "writable_attention_probs",
    "pytorch_fn_local", "generate",
})


def _ordered(values) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclasses.dataclass(frozen=True)
class InterventionSpec:
    """Backend-agnostic semantics shared by every family/backend cell for one methodology.

    The first seven coordinates use CausaLab's vocabulary. ``extensions`` records semantics that
    its v1 documents cannot express (for example a write at every decode step). ``semantic_params``
    and ``realization_params`` partition nnbench task parameters without changing the cell call.
    """

    data_roles: tuple[str, ...] = ("base",)
    components: tuple[str, ...] = ()
    operations: tuple[str, ...] = ("read",)
    mechanisms: tuple[str, ...] = ()
    position_frames: tuple[str, ...] = ("prompt",)
    featurizers: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    extensions: tuple[str, ...] = ()
    semantic_params: tuple[str, ...] = ()
    realization_params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Canonical ordering makes descriptors stable regardless of authoring order.
        for name in (
            "data_roles", "components", "operations", "mechanisms", "position_frames",
            "featurizers", "capabilities", "extensions", "semantic_params",
            "realization_params",
        ):
            object.__setattr__(self, name, _ordered(getattr(self, name)))

        bad_roles = [r for r in self.data_roles
                     if r not in {"base", "counterfactual"} and not r.startswith("counterfactual[")]
        checks = (
            (bad_roles, "data roles"),
            (set(self.components) - COMPONENTS, "components"),
            (set(self.operations) - OPERATIONS, "operations"),
            (set(self.mechanisms) - MECHANISMS, "write mechanisms"),
            (set(self.position_frames) - POSITION_FRAMES, "position frames"),
            (set(self.featurizers) - FEATURIZERS, "featurizers"),
            (set(self.capabilities) - CAPABILITIES, "capabilities"),
        )
        for bad, label in checks:
            if bad:
                raise ValueError(f"unknown CausaLab {label}: {sorted(bad)}")
        if "write" not in self.operations and self.mechanisms:
            raise ValueError("write mechanisms require the write operation")
        if "generated" in self.position_frames and "generate" not in self.capabilities:
            raise ValueError("the generated position frame requires the generate capability")
        overlap = set(self.semantic_params) & set(self.realization_params)
        if overlap:
            raise ValueError(f"params cannot be both semantic and realization coordinates: {sorted(overlap)}")

    def classify(self, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split cell params into protocol semantics and nnbench realization coordinates."""
        semantic_keys, realization_keys = set(self.semantic_params), set(self.realization_params)
        unknown = set(params) - semantic_keys - realization_keys
        if unknown:
            raise ValueError(f"unclassified task params for intervention spec: {sorted(unknown)}")
        semantics = {k: v for k, v in params.items() if k in semantic_keys}
        realization = {k: v for k, v in params.items() if k in realization_keys}
        return semantics, realization

    def coordinate(self) -> dict[str, list[str]]:
        """JSON-safe protocol coordinates for provenance and catalog indexing.

        This is deliberately not CausaLab's canonical document form. Digests and artifact identity
        remain owned by CausaLab; nnbench only publishes the semantic coordinates it measured.
        """
        return {
            "data_roles": list(self.data_roles),
            "components": list(self.components),
            "operations": list(self.operations),
            "mechanisms": list(self.mechanisms),
            "position_frames": list(self.position_frames),
            "featurizers": list(self.featurizers),
            "capabilities": list(self.capabilities),
            "extensions": list(self.extensions),
            "semantic_params": list(self.semantic_params),
        }


def _p(**kwargs) -> InterventionSpec:
    return InterventionSpec(**kwargs)


# One semantic description per methodology. Family-specific module paths and every spelling below
# this vocabulary remain in the explicit cells. This table is metadata, never an executor registry.
PROTOCOLS: dict[str, InterventionSpec] = {
    "logit_lens": _p(
        components=("block_output", "lm_head"), operations=("read",),
        capabilities=("full_logits",), semantic_params=("layers",),
        realization_params=("unembed", "residual"),
    ),
    "steering": _p(
        components=("block_output", "lm_head"), operations=("read", "write"),
        mechanisms=("add_scaled",), capabilities=("full_logits",),
        semantic_params=("layer", "target", "alpha"), realization_params=("mode",),
    ),
    "ablation": _p(
        components=("attention_output", "mlp_output", "block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",), capabilities=("full_logits",),
        semantic_params=("layer", "target"),
    ),
    "activation_patching": _p(
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",),
        capabilities=("paired_forward", "full_logits"), semantic_params=("layer", "patch"),
        realization_params=("residual",),
    ),
    "gen_patching": _p(
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",), position_frames=("prompt", "generated"),
        capabilities=("paired_forward", "full_logits", "generate"),
        semantic_params=("layer", "patch"), realization_params=("residual", "bound"),
    ),
    "gen_steering": _p(
        components=("block_output", "lm_head"), operations=("read", "write"),
        mechanisms=("add_scaled",), position_frames=("prompt", "generated"),
        capabilities=("full_logits", "generate"), extensions=("decode_step_write",),
        semantic_params=("layer", "target", "alpha"), realization_params=("bound",),
    ),
    "attention_pattern": _p(
        components=("attention_probs",), operations=("read",), semantic_params=("layers",),
    ),
    "attribution_patching": _p(
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "grad"), capabilities=("grad",), extensions=("activation_grad",),
        semantic_params=("grad",), realization_params=("residual",),
    ),
    "das": _p(
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "write", "grad"), mechanisms=("swap",), featurizers=("subspace",),
        capabilities=("grad", "paired_forward", "full_logits"), semantic_params=("train",),
    ),
    "jacobian_collect": _p(
        components=("block_output", "lm_head"), operations=("read", "grad"),
        capabilities=("grad",), extensions=("activation_grad", "jacobian_export"),
        semantic_params=("grad",),
    ),
    "jacobian_lens": _p(
        components=("block_output", "lm_head"), operations=("read",),
        capabilities=("full_logits",), extensions=("linear_transport",),
        semantic_params=("transport", "layers"), realization_params=("unembed",),
    ),
}


def protocol_for(methodology: str) -> InterventionSpec | None:
    return PROTOCOLS.get(methodology)

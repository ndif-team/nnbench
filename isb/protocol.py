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
    """Semantic descriptor, used as a methodology template or a concrete task description.

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
        semantic_params=("layer", "target", "alpha"), realization_params=("mode", "residual"),
    ),
    "ablation": _p(
        components=("attention_output", "mlp_output", "block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",), capabilities=("full_logits",),
        semantic_params=("layer", "target"), realization_params=("residual",),
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
        semantic_params=("layer", "patch", "new_tokens"), realization_params=("residual", "bound"),
    ),
    "gen_steering": _p(
        components=("block_output", "lm_head"), operations=("read", "write"),
        mechanisms=("add_scaled",), position_frames=("prompt", "generated"),
        capabilities=("full_logits", "generate"), extensions=("decode_step_write",),
        semantic_params=("layer", "target", "alpha", "new_tokens"), realization_params=("bound",),
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
        capabilities=("grad", "paired_forward", "full_logits"),
        semantic_params=("layer", "k", "train", "heldout", "lr", "seed"),
        realization_params=("residual",),
    ),
    "jacobian_collect": _p(
        components=("block_output",), operations=("read", "grad"),
        capabilities=("grad",), extensions=("activation_grad", "jacobian_export"),
        semantic_params=("grad", "skip_first"), realization_params=("dim_batch", "residual"),
    ),
    "jacobian_lens": _p(
        components=("block_output", "lm_head"), operations=("read",),
        capabilities=("full_logits",), extensions=("linear_transport",),
        semantic_params=("transport", "layers", "position"), realization_params=("unembed", "residual"),
    ),
}


def protocol_for(methodology: str) -> InterventionSpec | None:
    return PROTOCOLS.get(methodology)


def describe_task(methodology: str, params: Mapping[str, Any], *,
                  template: InterventionSpec, family: str) -> InterventionSpec:
    """Specialize metadata to the explicit cell's branches, without executing or routing it.

    Defaults here mirror the cells; regression tests exercise their branch behavior. Data roles
    describe the input contract even when a no-op case only executes one side of a paired feed.
    Call with effective params (including dataset defaults) when describing a bound run.
    """
    operations = set(template.operations)
    capabilities = set(template.capabilities)
    mechanisms = template.mechanisms
    extensions = set(template.extensions)
    components = template.components
    if ((methodology == "das" and not params.get("train", 0))
            or (methodology in {"attribution_patching", "jacobian_collect"}
                and not params.get("grad", True))):
        operations.discard("grad")
        capabilities.discard("grad")
        extensions.difference_update({"activation_grad", "jacobian_export"})
    no_write = (
        methodology in {"activation_patching", "gen_patching"} and not params.get("patch", True)
        or methodology == "steering" and params.get("alpha", 6.0) == 0
        or methodology == "ablation" and params.get("target") == "none"
    )
    if no_write:
        operations.discard("write")
        capabilities.discard("paired_forward")
        mechanisms = ()
        # Unpatched generation reads engine logits only; ordinary forward readouts also read
        # the final block residual. gen_steering still performs a write when alpha=0.
        components = (("lm_head",) if methodology == "gen_patching"
                      else ("block_output", "lm_head"))
    elif methodology == "ablation":
        target = params.get("target", "mixer" if family == "nemotron" else "mlp")
        component = {"attn": "attention_output", "mlp": "mlp_output"}.get(target)
        if component is not None:
            components = (component, "block_output", "lm_head")
        # Nemotron's mixer target is layer-dependent; retain the template's component union.
    if methodology == "jacobian_lens" and params.get("transport") is None:
        extensions.discard("linear_transport")
    return dataclasses.replace(template, operations=tuple(operations),
                               capabilities=tuple(capabilities), mechanisms=mechanisms,
                               extensions=tuple(extensions), components=components)

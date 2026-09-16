"""CausaLab-aligned semantic indexing for nnbench cells.

This torch-free module holds intervention descriptors and parameter classifications. Concrete
case requirements live beside cells in ``isb.methodologies.requirements``.
Explicit ``@cell(methodology, family, backend)`` functions implement the programs
whose backend realizations nnbench measures.

``InterventionSpec`` is the intersection of CausaLab's intervention-protocol vocabulary and the
coordinates nnbench needs to index a cell.  It is intentionally smaller than a runnable CausaLab
document: model and dataset bindings come from ``CellConfig``/``ExecutionRegime``; metrics, saves,
artifacts, and workflows remain CausaLab concerns.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_VOCABULARY = json.loads(Path(__file__).with_name("causalab_vocabulary.json").read_text())
CAUSALAB_REFERENCE = {key: _VOCABULARY[key] for key in ("repository", "revision")}
COMPONENTS = frozenset(_VOCABULARY["components"])
DEPRECATED_COMPONENTS = _VOCABULARY["deprecated_components"]
OPERATIONS = frozenset({"read", "write", "grad"})
MECHANISMS = frozenset(_VOCABULARY["mechanisms"])
POSITION_FRAMES = frozenset({"prompt", "generated"})
FEATURIZERS = frozenset(_VOCABULARY["featurizers"])
CAPABILITIES = frozenset(_VOCABULARY["capabilities"])


def _ordered(values) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclasses.dataclass(frozen=True)
class InterventionSpec:
    """Semantic descriptor, used as a methodology template or a concrete task description.

    Component, mechanism, featurizer and capability names use the pinned CausaLab vocabulary.
    Operations and frames summarize nnbench execution. ``extensions`` records semantics that
    upstream documents cannot express (for example a write at every decode step). ``semantic_params``
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
    # None preserves older/custom descriptors whose write targets were unspecified.
    write_components: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(
            DEPRECATED_COMPONENTS.get(c, c) for c in self.components))
        if self.write_components is not None:
            object.__setattr__(self, "write_components", _ordered(
                DEPRECATED_COMPONENTS.get(c, c) for c in self.write_components))
            if set(self.write_components) - set(self.components):
                raise ValueError("write components must belong to the descriptor's components")
            if self.write_components and "write" not in self.operations:
                raise ValueError("write components require the write operation")
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

    def coordinate(self) -> dict[str, Any]:
        """JSON-safe protocol coordinates for provenance and catalog indexing.

        These coordinates describe the measured case. The Docker job contract owns nnbench's
        experiment and artifact checksums; CausaLab owns its canonical document identities.
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
            "write_components": (list(self.write_components)
                                 if self.write_components is not None else None),
            "required_capabilities": self.required_capabilities(),
            "vocabulary": dict(CAUSALAB_REFERENCE),
        }

    def required_capabilities(self) -> list[str] | None:
        """Describe component requirements in CausaLab's engine vocabulary.

        Legacy write descriptors with unspecified targets have incomplete requirements.
        These requirements describe nnbench cells; backend selection remains explicit.
        """
        if "write" in self.operations and self.write_components is None:
            return None
        required = set(self.capabilities)
        required.update(f"component:{c}" for c in self.components)
        required.update(f"component:{c}:write" for c in self.write_components or ())
        return sorted(required)


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
        write_components=("block_output",),
        components=("block_output", "lm_head"), operations=("read", "write"),
        mechanisms=("add_scaled",), capabilities=("full_logits",),
        semantic_params=("layer", "target", "alpha"), realization_params=("mode", "residual"),
    ),
    "ablation": _p(
        write_components=("attention_output", "mlp_output"),
        components=("attention_output", "mlp_output", "block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",), capabilities=("full_logits",),
        semantic_params=("layer", "target"), realization_params=("residual",),
    ),
    "activation_patching": _p(
        write_components=("block_output",),
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",),
        capabilities=("paired_forward", "full_logits"), semantic_params=("layer", "patch"),
        realization_params=("residual",),
    ),
    "gen_patching": _p(
        write_components=("block_output",),
        data_roles=("base", "counterfactual"), components=("block_output", "lm_head"),
        operations=("read", "write"), mechanisms=("swap",), position_frames=("prompt", "generated"),
        capabilities=("paired_forward", "full_logits", "generate"),
        semantic_params=("layer", "patch", "new_tokens"), realization_params=("residual", "bound"),
    ),
    "gen_steering": _p(
        write_components=("block_output",),
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
        write_components=("block_output",),
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

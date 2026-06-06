"""Workload spec schema (design.md §11.2).

The spec is DATA: family-independent, serializable. It names *logical* locations
(the §11.3 vocabulary), never concrete module paths. Plain dataclasses (no pydantic
dependency) so the pure-logic layers import in any environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class Access:
    """Access kinds (design.md §11.2 `Selector.access`)."""

    READ = "read"
    WRITE_REPLACE = "write_replace"
    WRITE_INPLACE = "write_inplace"
    CACHE = "cache"
    GRAD = "grad"


ACCESS_KINDS = {
    Access.READ,
    Access.WRITE_REPLACE,
    Access.WRITE_INPLACE,
    Access.CACHE,
    Access.GRAD,
}


class AppState:
    """Applicability states (design.md §8.1) — the multi-valued cell of the map."""

    SUPPORTED = "SUPPORTED"
    SUPPORTED_DEGRADED = "SUPPORTED_DEGRADED"
    ERROR = "ERROR"
    SILENTLY_WRONG = "SILENTLY_WRONG"
    HANG = "HANG"
    UNSUPPORTED_BY_CONSTRUCTION = "UNSUPPORTED_BY_CONSTRUCTION"
    NO_REFERENCE = "NO_REFERENCE"  # ran, but the reference cell failed -> can't judge


# Logical target vocabulary (design.md §11.3). Tier annotations in comments.
TARGETS = {
    "block.output",   # a
    "block.input",    # a
    "attn.output",    # a
    "mlp.output",     # a
    "attn.head_value",  # b
    "mlp.neuron",       # b
    "attn.weights",     # c  (HF-eager only; frontier marker on vLLM)
    "logits",           # runtime
    "sampled_token",    # runtime
}


@dataclass
class Selector:
    """A logical address. Resolves (per family) to a set of concrete bindings."""

    target: str
    scope: Any = "all"        # "all" | list[int] | {start,stop,step} | {fraction: f}
    head: Any = None          # tier (b): int | "all" | list[int] | None
    neuron: Any = None        # tier (b): int | "all" | list[int] | None
    position: Any = "all"     # "all" | "last" | int | list[int]
    access: str = Access.READ

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"unknown target {self.target!r}; valid: {sorted(TARGETS)}")
        if self.access not in ACCESS_KINDS:
            raise ValueError(f"unknown access {self.access!r}; valid: {sorted(ACCESS_KINDS)}")


@dataclass
class AuxSpec:
    """A side module applied inside the trace (unembed / sae / probe)."""

    kind: str
    trainable: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class Inputs:
    kind: str = "single"               # single | list | dataset
    prompts: Optional[list] = None
    dataset: Optional[str] = None
    chat: bool = False
    pairs: bool = False                # counterfactual clean/corrupted pairs (patching)


@dataclass
class Generation:
    new_tokens: int = 0                # 0 == single forward pass
    per_step: bool = False             # intervene each decode step


@dataclass
class Workload:
    """One benchmark unit (design.md §11.2). Serialized as YAML on disk."""

    id: str
    motif: str
    tier: str = "method"               # micro | method | macro
    profile: str = "interactive"       # interactive | batched | generation | harvesting | concurrent
    selectors: list = field(default_factory=list)
    aux: list = field(default_factory=list)
    inputs: Inputs = field(default_factory=Inputs)
    generation: Generation = field(default_factory=Generation)
    params: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)  # backend_name -> AppState (missing => predicted)

    @staticmethod
    def from_dict(d: dict) -> "Workload":
        d = dict(d)
        d["selectors"] = [Selector(**s) for s in d.get("selectors", [])]
        d["aux"] = [AuxSpec(**a) for a in d.get("aux", [])]
        if "inputs" in d and isinstance(d["inputs"], dict):
            d["inputs"] = Inputs(**d["inputs"])
        if "generation" in d and isinstance(d["generation"], dict):
            d["generation"] = Generation(**d["generation"])
        return Workload(**d)


def load_yaml(path: str) -> Workload:
    """Load a workload spec from YAML (the on-disk 'dataset' form)."""
    import yaml  # optional dependency; only needed for YAML specs

    with open(path) as f:
        return Workload.from_dict(yaml.safe_load(f))

"""Resolver — the ONLY model-aware code (design.md §11.5).

Turns a logical Selector into a set of concrete `Binding`s by walking the model
with the FamilyProfile's path templates. Motif builders consume Bindings and never
see a path, which makes "don't hardcode to GPT-2/Llama" structurally impossible to
violate. The slice/reshape math (head/neuron/GQA) lives once here, parameterized by
`profile.dims`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..spec.schema import AppState, Selector
from .profiles import BackendProfile, FamilyProfile


@dataclass
class Binding:
    """A resolved concrete site (design.md §11.5)."""

    site_id: str
    module: Any                       # resolved nnsight Envoy (or any obj with .output)
    output_index: Optional[int]       # tuple-output index hint (None => not a tuple)
    reshape: Optional[tuple]          # e.g. (B, S, n_heads, head_dim) for per-head
    index: Optional[tuple]            # slice into the reshaped tensor (head/neuron/pos)
    access: str


class Unsupported(Exception):
    """Raised when a family cannot realize a requested logical target."""


class Resolver:
    def __init__(self, profile: FamilyProfile, model: Any):
        self.p = profile
        self.model = model
        self.config = getattr(model, "config", None)

    def capabilities(self) -> set:
        return self.p.caps

    def n_layers(self) -> int:
        return int(getattr(self.config, self.p.n_layers_attr))

    def _dim(self, name: str) -> Optional[int]:
        attr = self.p.dims.get(name)
        if attr is None:
            return None
        return int(getattr(self.config, attr))

    # --- path walking -----------------------------------------------------------

    def _walk(self, template: str, i: Optional[int] = None) -> Any:
        path = template.format(i=i) if i is not None else template
        obj = self.model
        for seg in path.split("."):
            if seg.lstrip("-").isdigit():
                obj = obj[int(seg)]
            else:
                obj = getattr(obj, seg)
        return obj

    def _scope_to_layers(self, scope: Any) -> list:
        n = self.n_layers()
        if scope == "all":
            return list(range(n))
        if isinstance(scope, (list, tuple)):
            return [i if i >= 0 else n + i for i in scope]
        if isinstance(scope, dict):
            if "fraction" in scope:
                k = max(1, int(round(n * float(scope["fraction"]))))
                step = max(1, n // k)
                return list(range(0, n, step))[:k]
            start = scope.get("start", 0)
            stop = scope.get("stop", n)
            step = scope.get("step", 1)
            return list(range(start, stop, step))
        raise ValueError(f"bad scope {scope!r}")

    # --- resolution -------------------------------------------------------------

    def resolve(self, sel: Selector) -> list:
        if sel.target not in self.p.caps:
            raise Unsupported(f"{self.p.family} cannot realize {sel.target}")
        role = self.p.layer_path_role(sel.target)
        out_idx = self.p.output_index.get(
            "block" if sel.target.startswith("block") else role
        )
        bindings = []
        for i in self._scope_to_layers(sel.scope):
            module = self._walk(self.p.paths[role], i)
            reshape, index = self._subblock(sel)
            bindings.append(
                Binding(
                    site_id=f"L{i}.{sel.target}"
                    + (f".h{sel.head}" if sel.head is not None else "")
                    + (f".n{sel.neuron}" if sel.neuron is not None else ""),
                    module=module,
                    output_index=out_idx,
                    reshape=reshape,
                    index=index,
                    access=sel.access,
                )
            )
        return bindings

    def _subblock(self, sel: Selector):
        """Compute (reshape, static-index) for tier-(b) head/neuron selectors.

        reshape is a partial shape with -1 placeholders for runtime (batch, seq);
        read_value fills those from the live tensor. Tier (a) returns (None, None).
        """
        if sel.target == "attn.head_value" and sel.head not in (None, "all"):
            head_dim = self._dim("head_dim")
            if head_dim is None:
                # derive from hidden size if not explicit
                n_heads = self._dim("n_heads")
                head_dim = None if n_heads is None else None  # filled at runtime
            return ("per_head", ("head", sel.head, head_dim))
        if sel.target == "mlp.neuron" and sel.neuron not in (None, "all"):
            return (None, ("neuron", sel.neuron))
        return (None, None)

    def resolve_one(self, role: str) -> Binding:
        """Resolve a singleton module (final_norm, unembed) — no layer index."""
        module = self._walk(self.p.paths[role])
        return Binding(
            site_id=role,
            module=module,
            output_index=None,
            reshape=None,
            index=None,
            access="read",
        )


def predict(
    workload, family: FamilyProfile, backend: BackendProfile, motif_requires=frozenset()
) -> str:
    """A-priori applicability cell (design.md §11.6): need ⊆ (family ∩ backend)?"""
    need = set(motif_requires)
    for s in workload.selectors:
        need.add(s.target)
        need.add(s.access)
    have = family.caps & backend.caps
    return AppState.SUPPORTED if need <= have else AppState.UNSUPPORTED_BY_CONSTRUCTION


def read_value(b: Binding):
    """Translate a Binding into an nnsight read: .output, untuple, reshape, slice."""
    out = b.module.output
    if isinstance(out, tuple):
        out = out[b.output_index if b.output_index is not None else 0]
    if b.reshape == "per_head" and b.index is not None:
        _, head, head_dim = b.index
        shp = out.shape
        if head_dim is None:
            # infer head_dim from a known n_heads if available; else leave as-is
            head_dim = shp[-1]
        view = out.view(shp[0], shp[1], shp[-1] // head_dim, head_dim)
        return view[:, :, head]
    if b.index is not None and b.index[0] == "neuron":
        return out[..., b.index[1]]
    return out

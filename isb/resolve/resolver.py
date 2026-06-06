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
    reshape: Optional[tuple]          # reserved for future explicit reshape
    index: Optional[tuple]            # tier-(b) slice tag: ("head",h,head_dim) | ("neuron",j)
    access: str
    side: str = "output"             # "output" (.output) or "input" (.input); §11.3 head_value


class Unsupported(Exception):
    """Raised when a family cannot realize a requested logical target."""


class Resolver:
    def __init__(self, profile: FamilyProfile, model: Any):
        self.p = profile
        self.model = model
        self.config = model.config   # a model always has .config; fail loudly if not

    def capabilities(self) -> set:
        return self.p.caps

    def n_layers(self) -> int:
        return int(getattr(self.config, self.p.n_layers_attr))

    def _dim(self, name: str) -> Optional[int]:
        attr = self.p.dims.get(name)
        val = getattr(self.config, attr) if attr is not None else None
        if val is None and name == "ffn":
            # GPT-2's n_inner defaults to None; derive 4*hidden from the profile's hidden attr.
            hidden_attr = self.p.dims.get("hidden")
            hidden = getattr(self.config, hidden_attr) if hidden_attr is not None else None
            return 4 * int(hidden) if hidden is not None else None
        return None if val is None else int(val)

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
        # §11.3: per-head value = o_proj/c_proj INPUT; residual-pre = block .input.
        side = "input" if sel.target in ("block.input", "attn.head_value") else "output"
        bindings = []
        for i in self._scope_to_layers(sel.scope):
            module = self._walk(self.p.paths[role], i)
            index = self._subblock(sel)
            bindings.append(
                Binding(
                    site_id=f"L{i}.{sel.target}"
                    + (f".h{sel.head}" if sel.head is not None else "")
                    + (f".n{sel.neuron}" if sel.neuron is not None else ""),
                    module=module,
                    output_index=out_idx,
                    reshape=None,
                    index=index,
                    access=sel.access,
                    side=side,
                )
            )
        return bindings

    def _subblock(self, sel: Selector):
        """Static slice tag for tier-(b) head/neuron selectors; None for tier (a).

        head_dim may be None (e.g. GPT-2 has no explicit head_dim) -> read_value
        derives it from the live tensor's last dim. Tier (a) returns None.
        """
        if sel.target == "attn.head_value" and sel.head not in (None, "all"):
            return ("head", sel.head, self._dim("head_dim"))
        if sel.target == "mlp.neuron" and sel.neuron not in (None, "all"):
            return ("neuron", sel.neuron)
        return None

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
    """A-priori applicability cell (design.md §11.6): need ⊆ (family ∩ backend)?

    `motif_requires` is the motif's capability requirement set. Per the layering in
    §11.10 (motifs depend on resolve, not vice-versa), the RUNNER looks it up from the
    motif registry and passes it here — this function stays registry-agnostic.
    """
    need = set(motif_requires)
    for s in workload.selectors:
        need.add(s.target)
        need.add(s.access)
    have = family.caps & backend.caps
    return AppState.SUPPORTED if need <= have else AppState.UNSUPPORTED_BY_CONSTRUCTION


def read_value(b: Binding):
    """Translate a Binding into an nnsight read: pick .input/.output, untuple, slice.

    Dispatch for tier-(b) is unified on the index tag (`("head",…)` / `("neuron",…)`).
    For `attn.head_value` the source is the o_proj/c_proj INPUT (§11.3), not its output.
    """
    src = b.module.input if b.side == "input" else b.module.output
    if isinstance(src, tuple):
        src = src[b.output_index if b.output_index is not None else 0]
    if b.index is None:
        return src
    tag = b.index[0]
    if tag == "head":
        _, head, head_dim = b.index
        shp = src.shape
        hd = head_dim if head_dim is not None else shp[-1]   # derive when not explicit
        return src.view(shp[0], shp[1], shp[-1] // hd, hd)[:, :, head]
    if tag == "neuron":
        return src[..., b.index[1]]
    return src

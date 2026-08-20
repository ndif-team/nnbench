"""Declarative sweep spec (design.md §12) — one CellConfig per methodology.

The 5 near-identical `scripts/smoke_*.py` collapse to one CellConfig each: the hardcoded
METHOD/FAMILY/REPO/PROMPTS/TASKS plus the per-script effect-size baseline become data here, and the
single execute layer (`isb/sweep/execute.py`) consumes them. No Resolver / YAML — the spec is Python next to
the cells, matching the flat `@cell` registry idiom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Workload:
    """An input regime. Batching is a coverage axis (it can change correctness), so each workload is
    oracle-checked in its own regime, not just timed.

    `prompts` is either a literal unit list (the bespoke single-trace specs) or a `DataRef` naming a
    registered source (isb/data.py) — data is a replaceable feed, so the same procedure runs any
    compatible dataset. A DataRef resolves at construction: its units land in `.prompts` and its
    per-dataset cell-param defaults in `.data_knobs` (injected UNDER task params by the driver, so
    an explicit task param always wins)."""
    kind: str                 # "interactive" (N independent prompts) | "batched" (N prompts) |
                              # "generation" (greedy multi-token decode; cells read/intervene per step)
    prompts: object           # list of units, or a DataRef
    new_tokens: int = 0       # generation: decode steps per prompt; injected into cell params by the
                              # driver (the regime axis lives on the Workload, not in every task dict)
    aggregate: bool = True    # interactive/generation: run each prompt as its OWN trace and score the
                              # verdict aggregated over all of them (top-1 fraction + mean TV) — robust,
                              # not a single-token anecdote. Set False for cells that consume their
                              # prompt list as ONE unit (clean/corrupt pairs) or can't stack (attention
                              # maps).
    data_knobs: dict = field(default_factory=dict)
    data_name: str | None = None

    def __post_init__(self):
        from ..data import DataRef, load_data

        if isinstance(self.prompts, DataRef):
            ref = self.prompts
            self.prompts, knobs = load_data(ref)
            self.data_knobs = {**knobs, **self.data_knobs}
            self.data_name = ref.name
        if self.kind == "generation" and self.new_tokens <= 0:
            raise ValueError("generation workload needs new_tokens>0")
        if self.kind not in ("interactive", "batched", "generation"):
            raise ValueError(f"workload kind {self.kind!r} not implemented in v1")
        if self.aggregate and self.kind == "batched":
            self.aggregate = False    # batched runs ONE padded trace by definition; never per-prompt


@dataclass
class BaselineSpec:
    """The no-intervention reference run on the SAME backend — the overhead denominator. It is the
    same methodology cell with no-op params (e.g. steering alpha=0, ablation target='none', logit
    lens reading its own portable form), so no extra code per methodology."""
    params: dict
    label: str = "baseline"


@dataclass
class EffectSpec:
    """Non-vacuity guard for write methodologies: the intervention must move the control's output
    (else a backend that drops the write scores SUPPORTED vacuously). TV(control baseline, control
    perturbed) must clear the floor (or top-1 must flip)."""
    baseline_params: dict
    perturbed_params: dict
    tv_floor: float = 0.2
    top1_ceiling: float = 0.5


@dataclass
class CellConfig:
    name: str                                   # spec id for the CLI: `bench.py --spec <name>`
    methodology: str
    family: str
    repo: str
    workloads: list                             # [Workload(...)]
    tasks: list                                 # [(params: dict, label: str), ...]
    baseline: BaselineSpec
    effect: Optional[EffectSpec] = None         # None for read methodologies (no write to guard)
    dtype_control: str = "float32"              # control precision for the SILENTLY_WRONG-vs-DEGRADED re-check
    warmup: int = 3
    n_trials: int = 7
    hf_kwargs: dict = field(default_factory=dict)
    vllm_kwargs: dict = field(default_factory=dict)


def spec_with_data(spec: CellConfig, ref) -> CellConfig:
    """The same procedure over a different data source (`bench.py --data`): interactive/generation
    workloads rebind to `ref`'s units; a batched workload rebuilds from the first 16 units of the
    same source (the padded-batch REGIME is the point there, not volume). Unit kinds must match —
    binding pair-data to a prompt-procedure (or vice versa) is a loud error. The copy's name gains
    an `@source` suffix so refs, outputs, and banners never collide with the default binding."""
    import dataclasses

    from ..data import load_data, unit_kind

    units, knobs = load_data(ref)
    rebound = []
    for w in spec.workloads:
        u0 = w.prompts[0] if w.prompts else None
        current = ("pair_labeled" if isinstance(u0, tuple) and len(u0) == 3
                   else "pair" if isinstance(u0, tuple) else "prompt")
        if unit_kind(ref.name) != current:
            raise ValueError(
                f"data source {ref.name!r} yields {unit_kind(ref.name)!r} units but "
                f"{spec.name!r}'s {w.kind} workload consumes {current!r} units")
        if w.kind == "batched":
            rebound.append(Workload("batched", units[:16],
                                    data_knobs=dict(knobs), data_name=ref.name))
        else:
            rebound.append(Workload(w.kind, list(units), new_tokens=w.new_tokens,
                                    aggregate=w.aggregate,
                                    data_knobs=dict(knobs), data_name=ref.name))
    return dataclasses.replace(
        spec, name=f"{spec.name}@{ref.name.replace('/', '-')}", workloads=rebound)

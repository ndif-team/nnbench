"""Declarative sweep spec (design.md §12) — one CellConfig per methodology.

The 5 near-identical `scripts/smoke_*.py` collapse to one CellConfig each: the hardcoded
METHOD/FAMILY/REPO/PROMPTS/TASKS plus the per-script effect-size baseline become data here, and the
single driver (`isb/sweep/driver.py`) consumes them. No Resolver / YAML — the spec is Python next to
the cells, matching the flat `@cell` registry idiom.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Workload:
    """An input regime. Batching is a coverage axis (it can change correctness), so each workload is
    oracle-checked in its own regime, not just timed."""
    kind: str                 # "interactive" (1 prompt) | "batched" (N prompts) | "generation" (stub)
    prompts: list
    new_tokens: int = 0       # generation stub; asserted == 0 in v1

    def __post_init__(self):
        if self.kind == "generation" and self.new_tokens == 0:
            raise ValueError("generation workload needs new_tokens>0 (v1 stub: not implemented)")
        if self.kind not in ("interactive", "batched"):
            raise ValueError(f"workload kind {self.kind!r} not implemented in v1")


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

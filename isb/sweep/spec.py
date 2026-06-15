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
    kind: str                 # "interactive" (N independent prompts) | "batched" (N prompts) |
                              # "generation" (greedy multi-token decode; cells read/intervene per step)
    prompts: list
    new_tokens: int = 0       # generation: decode steps per prompt; injected into cell params by the
                              # driver (the regime axis lives on the Workload, not in every task dict)
    aggregate: bool = True    # interactive/generation: run each prompt as its OWN trace and score the
                              # verdict aggregated over all of them (top-1 fraction + mean TV) — robust,
                              # not a single-token anecdote. Set False for cells that consume their
                              # prompt list as ONE unit (clean/corrupt pairs) or can't stack (attention
                              # maps).

    def __post_init__(self):
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
    # The benchmark's encoded knowledge: what each cell is EXPECTED to do, so a run reports the DELTA
    # ("cell X flipped") instead of restating the map. Keyed by (backend, workload_kind, label); only
    # the non-SUPPORTED cells need listing — anything unlisted defaults to SUPPORTED. A vllm_serve cell
    # with no entry inherits the vllm_async expectation (serve should match in-process vLLM; a mismatch
    # is a genuine transport surprise). See `expected_state` in the driver.
    #
    # CONVENTION (two modes, one field): an entry is the cell's known **steady state** on the bench's
    # reference nnsight — engine-mode-independent statuses are verified identical on dev and the fix
    # branch (e.g. the vllm_sync batched/interactive entries). The one deliberate exception is a
    # **flip-detector**: for a cell whose status an upstream fix *in flight* will change, the entry
    # holds the PRE-FIX baseline so the fix surfaces as a ⚠ SURPRISE on a fix-branch run (e.g.
    # gen_steering's unbounded `iter[:]` left at ERROR; the micro tier's barrier/iteration in
    # `isb/micro/probes.py`). Same regression-detector role, pointed at an expected change.
    expected: dict = field(default_factory=dict)

"""Benchmark definitions (design.md §12.16).

Authoring form: a ``Workload`` (methodology, semantic ``Row`` s, ``Regime`` s with default data,
baseline, ``Effect``) names no model and no system. ``instantiate(workload, model)`` resolves it on
one ``isb.models.Model`` into a ``CellConfig``, the resolved record the job contract freezes:
``InterventionSpec`` describes the methodology, ``ExecutionRegime`` the inputs, and ``TaskSpec`` one
case and the params its cell receives.
"""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from ..protocol import InterventionSpec, protocol_for

REALIZATION_MODES = ("default", "all")


@dataclass
class ExecutionRegime:
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
                              # driver (the regime axis lives here, not in every task dict)
    aggregate: bool = True    # interactive/generation: run each prompt as its OWN trace and score the
                              # verdict aggregated over all of them (top-1 fraction + mean TV) — robust,
                              # not a single-token anecdote. Set False for cells that consume their
                              # prompt list as ONE unit (clean/corrupt pairs) or can't stack (attention
                              # maps).
    data_knobs: dict = field(default_factory=dict)
    data_name: str | None = None
    # Load patterns (design.md §12.17). offline: the units issued as one sweep, `batch_size` units
    # per call (None = every unit in one call). concurrent: `concurrency` clients each issuing one
    # unit per call until the units run out. open: one-unit requests arriving at `rate` per second
    # whatever the system does. All keep the outputs of the first `score_units` units for scoring;
    # the rest are timed only.
    batch_size: int | None = None
    concurrency: int = 0
    rate: float = 0.0
    score_units: int = 16
    passes: int = 1           # load patterns: timed sweeps over all units, after one warm-up

    def __post_init__(self):
        from ..data import DataRef, load_data

        if isinstance(self.prompts, DataRef):
            ref = self.prompts
            self.prompts, knobs = load_data(ref)
            self.data_knobs = {**knobs, **self.data_knobs}
            self.data_name = ref.name
        if self.kind == "generation" and self.new_tokens <= 0:
            raise ValueError("generation regime needs new_tokens>0")
        if self.kind not in REGIME_KINDS:
            raise ValueError(f"execution regime {self.kind!r} not implemented")
        if self.aggregate and self.kind in ("batched", "offline"):
            self.aggregate = False    # one call consumes several units; never per-prompt
        if self.kind == "offline" and self.batch_size is not None and self.batch_size < 1:
            raise ValueError("offline batch_size must be positive, or None for every unit at once")
        if self.kind == "concurrent" and self.concurrency < 1:
            raise ValueError("concurrent regime needs concurrency>=1")
        if self.kind == "open" and not (self.rate > 0 and _RATE.fullmatch(f"{self.rate:g}")):
            raise ValueError(f"open regime needs a positive rate written without an exponent, "
                             f"got {self.rate!r}")
        if self.kind in LOAD_KINDS and (self.score_units < 1 or self.passes < 1):
            raise ValueError("score_units and passes must be positive")

    @property
    def name(self) -> str:
        """The regime's key in outputs and records (see `regime_name`)."""
        return regime_name(self.kind, vars(self))


REGIME_KINDS = ("interactive", "batched", "generation", "offline", "concurrent", "open")
# ExecutionRegime fields each load pattern records; other regimes' records leave every load field
# out, and each load pattern leaves out the others', so adding a pattern changes no existing
# experiment identity.
LOAD_FIELDS = {
    "offline": ("batch_size", "concurrency", "score_units", "passes"),
    "concurrent": ("batch_size", "concurrency", "score_units", "passes"),
    "open": ("rate", "score_units", "passes"),
}
LOAD_KINDS = tuple(LOAD_FIELDS)
ALL_LOAD_FIELDS = tuple(dict.fromkeys(f for fields in LOAD_FIELDS.values() for f in fields))
_RATE = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]+)?")
_LOAD_NAME = re.compile(rf"(offline/(all|b[1-9][0-9]*)|concurrent/c[1-9][0-9]*|open/r{_RATE.pattern})")


def regime_name(kind: str, settings=None) -> str:
    """A regime's key: its kind, plus the load setting for a load pattern, so one run can hold
    several (`offline/b16`, `offline/all`, `concurrent/c4`, `open/r8`; design.md §12.17).
    `settings` maps the load fields (a regime record, its `load` object, or `vars(regime)`)."""
    settings = settings or {}
    if kind == "offline":
        batch_size = settings.get("batch_size")
        return f"offline/{'all' if batch_size is None else f'b{batch_size}'}"
    if kind == "concurrent":
        return f"concurrent/c{settings.get('concurrency', 0)}"
    if kind == "open":
        return f"open/r{settings.get('rate', 0):g}"
    return kind


def is_load_regime(name) -> bool:
    """Whether a regime key (ExecutionRegime.name) names a load pattern."""
    return isinstance(name, str) and _LOAD_NAME.fullmatch(name) is not None


def scored_regime(name) -> bool:
    """Whether an output/record key's first element names a scored regime."""
    return isinstance(name, str) and (name in ("interactive", "batched", "generation")
                                      or is_load_regime(name))


def reference_key(name: str, row: str, comparison: str = "correctness") -> tuple:
    """The reference output a candidate cell `(name, row)` is scored against. For correctness, a
    call that consumed several units at once (batched, offline) is scored against the reference's
    per-unit outputs, never its own multi-unit batch (left padding is part of what is checked);
    everything else, and every equivalence comparison, against the same regime's output."""
    if comparison == "correctness":
        if name == "batched":
            return ("batched_perprompt", row)
        if name.startswith("offline/"):
            return ("offline_perprompt", row)
    return (name, row)


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
class TaskSpec:
    """One benchmark case: a label and the params its cell receives. The run records the params a
    call actually ran with, split by the methodology's protocol into semantics and realization
    spelling (isb/sweep/execute.py); the spec stores only what the author wrote."""
    label: str
    params: dict = field(default_factory=dict)


@dataclass
class CellConfig:
    name: str                                   # `<workload>/<model>` (+ `@<data>`), generated
    methodology: str
    family: str
    repo: str
    regimes: list[ExecutionRegime]
    tasks: list                                 # [TaskSpec] or [(params, label)]; normalized to TaskSpec
    baseline: BaselineSpec
    effect: Optional[EffectSpec] = None         # None for read methodologies (no write to guard)
    dtype_control: str = "float32"              # control precision for the SILENTLY_WRONG-vs-DEGRADED re-check
    warmup: int = 3
    n_trials: int = 7
    hf_kwargs: dict = field(default_factory=dict)
    vllm_kwargs: dict = field(default_factory=dict)
    # checked against the loaded model when set (instantiate sets it from the model row)
    n_layers: int | None = None
    # "default": each row runs the system's documented realization only; "all": also every
    # alternative the system's cell registration declares (design.md §12.16)
    realizations: str = "default"
    protocol: InterventionSpec | None = None    # defaults to the methodology's PROTOCOLS entry
    protocol_absence_reason: str | None = None  # required when no protocol describes the methodology
    protocol_source: str = field(default="authored", compare=False, repr=False)

    def __post_init__(self):
        if self.realizations not in REALIZATION_MODES:
            raise ValueError(f"realizations must be one of {REALIZATION_MODES}")
        if self.protocol_source not in {"authored", "restored"}:
            raise ValueError(f"unknown protocol source {self.protocol_source!r}")
        if self.protocol_source == "authored" and self.protocol is None:
            self.protocol = protocol_for(self.methodology)
        self.protocol_coverage()
        self.tasks = [t if isinstance(t, TaskSpec) else TaskSpec(label=t[1], params=t[0])
                      for t in self.tasks]
        if self.protocol is not None and self.protocol_source == "authored":
            # Every param a spec passes must be one the protocol names, loud at import; baseline
            # and effect params are cell calls too.
            declared = [t.params for t in self.tasks] + [self.baseline.params]
            if self.effect is not None:
                declared += [self.effect.baseline_params, self.effect.perturbed_params]
            for params in declared:
                self.protocol.classify(params)

    def protocol_coverage(self) -> dict:
        """Validate and describe this spec's explicit metadata coverage policy."""
        if self.protocol is not None:
            if self.protocol_absence_reason is not None:
                raise ValueError("described methods cannot supply protocol_absence_reason")
            return {"status": "described"}
        if self.protocol_source == "authored" and protocol_for(self.methodology) is not None:
            raise ValueError(f"built-in methodology {self.methodology!r} requires a protocol")
        reason = self.protocol_absence_reason
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"methodology {self.methodology!r} requires a protocol descriptor "
                             "or an explicit protocol_absence_reason")
        return {"status": "undescribed", "reason": reason}


def spec_with_data(spec: CellConfig, ref) -> CellConfig:
    """The same procedure over a different data source (`bench.py --data`): interactive/generation
    regimes rebind to `ref`'s units; a batched regime rebuilds from the first 16 units of the
    same source (the padded-batch REGIME is the point there, not volume). Unit kinds must match —
    binding pair-data to a prompt-procedure (or vice versa) is a loud error. The copy's name gains
    an `@source` suffix so refs, outputs, and banners never collide with the default binding."""
    import dataclasses

    from ..data import load_data, unit_kind

    units, knobs = load_data(ref)
    rebound = []
    for w in spec.regimes:
        u0 = w.prompts[0] if w.prompts else None
        current = ("pair_labeled" if isinstance(u0, tuple) and len(u0) == 3
                   else "pair" if isinstance(u0, tuple) else "prompt")
        if unit_kind(ref.name) != current:
            raise ValueError(
                f"data source {ref.name!r} yields {unit_kind(ref.name)!r} units but "
                f"{spec.name!r}'s {w.kind} regime consumes {current!r} units")
        if w.kind == "batched":
            rebound.append(ExecutionRegime("batched", units[:16],
                                           data_knobs=dict(knobs), data_name=ref.name))
        elif w.kind in LOAD_KINDS:                    # a load pattern keeps its unit count
            rebound.append(ExecutionRegime(w.kind, list(units[:len(w.prompts)]),
                                           new_tokens=w.new_tokens, aggregate=w.aggregate,
                                           data_knobs=dict(knobs), data_name=ref.name,
                                           batch_size=w.batch_size, concurrency=w.concurrency,
                                           rate=w.rate, score_units=w.score_units, passes=w.passes))
        else:
            rebound.append(ExecutionRegime(w.kind, list(units), new_tokens=w.new_tokens,
                                           aggregate=w.aggregate,
                                           data_knobs=dict(knobs), data_name=ref.name))
    return dataclasses.replace(
        spec, name=f"{spec.name}@{ref.name.replace('/', '-')}", regimes=rebound)


# ---- workloads (design.md §12.16) ---------------------------------------------------------------

@dataclass(frozen=True)
class Depth:
    """A layer at fraction `f` of the model's depth: `round(f * n_layers)`, clamped to the last layer."""
    f: float

    def __post_init__(self):
        if not 0.0 <= self.f <= 1.0:
            raise ValueError("depth fraction must be within [0, 1]")

    def resolve(self, n_layers: int) -> int:
        return min(int(round(self.f * n_layers)), n_layers - 1)


@dataclass(frozen=True)
class Asset:
    """A field of a fitted artifact bound to the model (`Model.assets[name][key]`). A row that
    references one requires the model to carry that asset."""
    name: str
    key: str


@dataclass(frozen=True)
class Row:
    """One workload case: a label saying what it does, semantic params, and the block components
    it needs beyond what its params reference (`attn`, `mlp`, `mixer`)."""
    label: str
    params: dict = field(default_factory=dict)
    requires: tuple = ()


@dataclass(frozen=True)
class Regime:
    """An input regime template; `data` is a DataRef or a literal unit list."""
    kind: str
    data: object
    new_tokens: int = 0
    aggregate: bool = True


@dataclass(frozen=True)
class Offline:
    """MLPerf Offline / `vllm bench throughput` (design.md §12.17): `units` units of the workload's
    batched regime's data issued as one sweep, `batch_size` units per call (None = every unit in
    one call, the engine schedules them). One regime per entry of `batch_sizes`."""
    units: int
    batch_sizes: tuple = (None,)
    passes: int = 3


@dataclass(frozen=True)
class Concurrent:
    """Closed-loop serving (`vllm bench serve --max-concurrency`, AIPerf's concurrency sweep;
    design.md §12.17): `units` units of the workload's single-unit regime (interactive, else
    generation), with `c` clients each issuing one unit per call until none are left. One regime
    per entry of `concurrency`."""
    units: int
    concurrency: tuple = (1,)
    passes: int = 1           # one sweep of `units` requests gives the latency distribution


@dataclass(frozen=True)
class Open:
    """Open-loop serving (MLPerf Server, `vllm bench serve --request-rate`; design.md §12.17):
    `units` units of the workload's single-unit regime arriving as a seeded Poisson process at
    `rate` requests per second, whatever the system is doing. One regime per entry of `rates`."""
    units: int
    rates: tuple = (1.0,)
    passes: int = 1


@dataclass(frozen=True)
class Effect:
    """The non-vacuity guard. `perturbed` lists candidate rows in order; the first one the model
    can express is used."""
    baseline: dict
    perturbed: tuple
    tv_floor: float = 0.2
    top1_ceiling: float = 0.5


@dataclass(frozen=True)
class Workload:
    name: str
    methodology: str
    rows: tuple
    regimes: tuple
    baseline: dict
    effect: Optional[Effect] = None
    warmup: int = 3
    n_trials: int = 7
    # The load patterns (LOAD_KINDS) that match how this methodology is used: offline for dataset
    # studies, concurrent and open for interventions on live traffic (design.md §12.17). A suite's
    # load applies only to the workloads that declare it.
    load_patterns: tuple = ()

    def __post_init__(self):
        if not set(self.load_patterns) <= set(LOAD_KINDS):
            raise ValueError(f"workload {self.name!r}: unknown load patterns "
                             f"{sorted(set(self.load_patterns) - set(LOAD_KINDS))}")
        # Workload params are semantics only; a realization belongs to the system (§12.16).
        protocol = protocol_for(self.methodology)
        declared = [row.params for row in self.rows] + [self.baseline]
        if self.effect is not None:
            declared += [self.effect.baseline] + [row.params for row in self.effect.perturbed]
        labels = [row.label for row in self.rows]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"workload {self.name!r}: row labels must be nonempty and unique")
        if protocol is None:
            return
        for params in declared:
            semantics, realization = protocol.classify(params)
            if realization:
                raise ValueError(f"workload {self.name!r} passes realization params {sorted(realization)}; "
                                 "declare them on the system's cell registration instead")


class NotApplicable(Exception):
    """The model cannot express this row (a component or asset it lacks)."""


def _resolve(value, model, needs: set):
    if isinstance(value, Depth):
        return value.resolve(model.n_layers)
    if isinstance(value, Asset):
        needs.add(f"asset:{value.name}")
        if value.name not in model.assets:
            raise NotApplicable(f"{model.key} has no {value.name!r} asset")
        return deepcopy(model.assets[value.name][value.key])
    if isinstance(value, dict):
        return {k: _resolve(v, model, needs) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, model, needs) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve(v, model, needs) for v in value)
    return value


def _check_layers(params, model):
    n = model.n_layers
    indices = [params["layer"]] if isinstance(params.get("layer"), int) else []
    if isinstance(params.get("layers"), list):
        indices += [i for i in params["layers"] if isinstance(i, int)]
    for i in indices:
        if not -n <= i < n:
            raise ValueError(f"layer {i} is out of range for {model.key} ({n} layers)")


def _bind(params, requires, model):
    from ..models import FAMILIES

    missing = sorted(set(requires) - FAMILIES[model.family].components)
    if missing:
        raise NotApplicable(f"{model.key} has no {', '.join(missing)} block component")
    resolved = _resolve(params, model, set())
    _check_layers(resolved, model)
    return resolved


def _regime(template, data=None, size=None):
    from ..data import DataRef

    source = data if data is not None else template.data
    if isinstance(source, DataRef) and size is not None:
        source = DataRef(source.name, min(size, 16) if template.kind == "batched" else size)
    return ExecutionRegime(template.kind, source, new_tokens=template.new_tokens,
                           aggregate=template.aggregate)


def _resized(data, units):
    from ..data import DataRef

    return DataRef(data.name, units) if isinstance(data, DataRef) else list(data)[:units]


def _load_regimes(workload: Workload, load):
    """The regimes a load pattern derives from `workload`'s own regimes, and the reason for any it
    leaves out (None when there is none). A load the workload does not declare derives nothing.
    Offline sweeps the batched regime's data, B units per call; a workload whose cells take one
    unit per call (an aggregate interactive, else generation, regime) sweeps at B=1 only.
    Concurrent and open issue that single-unit regime's units. The derived regime keeps the
    template's data source, resized to the load's unit count, and its decode length."""
    kind = {Offline: "offline", Concurrent: "concurrent", Open: "open"}.get(type(load))
    if kind is None:
        raise TypeError(f"unknown load pattern {load!r}")
    if kind not in workload.load_patterns:
        return [], (f"{kind} does not match how this workload is used (its load patterns: "
                    f"{', '.join(workload.load_patterns) or 'none'})")
    single = next((r for k in ("interactive", "generation") for r in workload.regimes
                   if r.kind == k and r.aggregate), None)
    if kind == "offline":
        batched = next((r for r in workload.regimes if r.kind == "batched"), None)
        template, sizes, reason = batched, load.batch_sizes, None
        if batched is None:
            # a workload that declares offline but whose only regime hands the whole unit list to
            # one call (Jacobian collection: the corpus mean of independent per-prompt maps) sweeps
            # that regime one unit per call
            template = single or next((r for r in workload.regimes if r.kind == "interactive"), None)
            if template is None:
                return [], "offline needs a batched or single-unit regime"
            sizes = tuple(b for b in load.batch_sizes if b == 1)
            if len(sizes) < len(load.batch_sizes):
                reason = "this workload's cells take one unit per call, so offline runs at B=1 only"
        return [ExecutionRegime("offline", _resized(template.data, load.units), batch_size=b,
                                new_tokens=template.new_tokens, passes=load.passes)
                for b in sizes], reason
    if single is None:
        return [], f"{kind} needs a single-unit interactive or generation regime"
    if kind == "concurrent":
        return [ExecutionRegime("concurrent", _resized(single.data, load.units),
                                new_tokens=single.new_tokens, concurrency=c, passes=load.passes)
                for c in load.concurrency], None
    return [ExecutionRegime("open", _resized(single.data, load.units), new_tokens=single.new_tokens,
                            rate=float(r), passes=load.passes)
            for r in load.rates], None


def instantiate(workload: Workload, model, *, data=None, size=None, regimes=None,
                realizations="default", warmup=None, n_trials=None, loads=()):
    """Resolve `workload` on `model` (an isb.models.Model or its key). Returns the CellConfig and
    the rows the model cannot express, as [{"label", "reason"}]. `data` (a DataRef) rebinds every
    regime's source, `size` resizes each regime's default source, `regimes` keeps a subset of kinds,
    and `loads` (Offline / Concurrent / Open) adds the load-pattern regimes derived from the workload's
    own (design.md §12.17); a load the workload cannot take is reported as a skipped row."""
    from ..models import MODELS

    if isinstance(model, str):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; available: {', '.join(MODELS)}")
        model = MODELS[model]
    tasks, skipped = [], []
    for row in workload.rows:
        try:
            tasks.append(TaskSpec(row.label, _bind(row.params, row.requires, model)))
        except NotApplicable as reason:
            skipped.append({"label": row.label, "reason": str(reason)})
    if not tasks:
        raise NotApplicable(f"{model.key} can express no row of {workload.name!r}: "
                            + "; ".join(s["reason"] for s in skipped))
    effect = None
    if workload.effect is not None:
        for candidate in workload.effect.perturbed:
            try:
                perturbed = _bind(candidate.params, candidate.requires, model)
            except NotApplicable:
                continue
            effect = EffectSpec(_bind(workload.effect.baseline, (), model), perturbed,
                                tv_floor=workload.effect.tv_floor,
                                top1_ceiling=workload.effect.top1_ceiling)
            break
        else:
            raise NotApplicable(f"{model.key} can express no effect check of {workload.name!r}")
    kept = [_regime(r, None, size) for r in workload.regimes if regimes is None or r.kind in regimes]
    for load in loads:
        derived, reason = _load_regimes(workload, load)
        if reason is not None:
            skipped.append({"label": f"[{type(load).__name__.lower()} load]", "reason": reason})
        kept += [r for r in derived if regimes is None or r.kind in regimes]
    if not kept:                # e.g. a load pattern the workload cannot take, and no other regime
        reasons = "".join(f"; {row['reason']}" for row in skipped if row["label"].endswith(" load]"))
        raise NotApplicable(f"workload {workload.name!r} has none of the regimes {sorted(regimes)}{reasons}")
    spec = CellConfig(
        name=f"{workload.name}/{model.key}", methodology=workload.methodology,
        family=model.family, repo=model.repo,
        regimes=kept,
        tasks=tasks, baseline=BaselineSpec(_bind(workload.baseline, (), model)), effect=effect,
        dtype_control=model.dtype_control,
        warmup=workload.warmup if warmup is None else warmup,
        n_trials=workload.n_trials if n_trials is None else n_trials,
        hf_kwargs=deepcopy(model.hf_kwargs), vllm_kwargs=deepcopy(model.vllm_kwargs),
        n_layers=model.n_layers, realizations=realizations)
    if data is not None:
        spec = spec_with_data(spec, data)
    return spec, skipped

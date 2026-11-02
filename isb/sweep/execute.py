"""Execute one run: (spec × data × RunConfig) -> one run file, <out_dir>/<run_name>.pt
(isb/runfile.py): `outputs` {(regime name, label): cpu tensor} holds every cell's warm output (an
alternative realization under its own label, `<row> [k=v]`), the per-unit stacks a multi-unit call
is scored against (("batched_perprompt" | "offline_perprompt", label)) and ("__meta__",), the
per-cell perf (with `returned_bytes`, the tensor bytes one call hands the client), errors and
effective params; `provenance` is the client/deployment/engine/host
record with the run's coordinates (isb/runs.py `run_coordinates`).

Every regime runs the same way: the baseline cell, the plain request (`be.plain`) and every case,
each over the regime's units in the regime's call form and timing loop (warm trials for the
execution regimes, the load pattern's passes for a load regime). Scoring is a separate step
(score.py) over run files: execution never compares.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import traceback
from copy import deepcopy
from dataclasses import dataclass, field

from ..methodologies.observations import capture_choices
from ..methodologies.registry import Unsupported, get_cell, get_realizations
from ..perf.core import returned_bytes
from ..perf.timing import time_cell
from ..perf.worker_memory import WorkerMemory, allow_callable_rpc, requested as worker_memory_requested
from ..runfile import save_run
from ..runs import RunConfig, cell_interface, make_backend, resolve_provenance, spec_coordinates
from ..sweep.guards import compute_effect_size
from ..sweep.spec import LOAD_KINDS


# ---- cell-call helpers (shared with tests) ----------------------------------------------------

def _cell(name, methodology, family):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise Unsupported(f"no cell for {methodology}/{family}/{name}")
    return fn


def _effective_params(fn, params):
    """Bind the standard cell arguments and apply Python defaults, including named **kwargs."""
    signature = inspect.signature(fn)
    positional = (object(), object(), object())
    leading = signature.bind_partial(*positional).arguments
    bound = signature.bind(*positional, **params)
    bound.apply_defaults()
    effective = {}
    for name, value in bound.arguments.items():
        if name in leading:
            continue
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            effective.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            effective[name] = value
    return deepcopy(effective)


@dataclass
class BoundCell:
    """One resolved call with owned settings; each invocation prepares fresh parameter values."""
    fn: object
    params: dict
    record: dict = field(default_factory=dict)
    positional_names: tuple = field(init=False)

    def __post_init__(self):
        parameters = list(inspect.signature(self.fn).parameters.values())
        self.positional_names = tuple(p.name for p in parameters[3:]
                                      if p.kind is inspect.Parameter.POSITIONAL_ONLY)

    def prepare(self, impl, model, units):
        """A fresh zero-argument call of the cell over `units` (a list)."""
        owned = deepcopy(self.params)
        positional = [owned.pop(name) for name in self.positional_names]

        def invoke():
            with capture_choices(self.params, self.record):
                return self.fn(impl, model, list(units), *positional, **owned)

        return invoke


def _bind_case(spec, name, params, record):
    """Populate the attempted/effective record before each fallible stage of call preparation."""
    from ..methodologies.requirements import describe_case

    record.update(attempted_params=deepcopy(params), error_stage="resolve",
                  protocol_coverage=deepcopy(spec.protocol_coverage()))
    fn = _cell(name, spec.methodology, spec.family)
    record["error_stage"] = "bind"
    effective = _effective_params(fn, params)
    record["params"] = deepcopy(effective)
    record["error_stage"] = "classify"
    if spec.protocol is not None:
        semantics, realization = spec.protocol.classify(effective)
        record.update(semantics=deepcopy(semantics), realization=deepcopy(realization))
    record["error_stage"] = "describe"
    record.update(deepcopy(describe_case(spec.methodology, deepcopy(effective),
                                        template=spec.protocol, family=spec.family)))
    record["error_stage"] = "execute"
    return BoundCell(fn, effective, record)


def _failure(record, error):
    if isinstance(error, Unsupported):     # the system lacks the capability
        record.update(unsupported=str(error), error=None, error_stage=None)
        return
    traceback.print_exc()
    record["error"] = repr(error)[:1000]


def _success(record):
    record.update(error=None, error_stage=None)


def _task_params(regime, params):
    """Per-call cell params = dataset knobs (per-source defaults, e.g. the upstream readout-position
    rule) under the task's explicit params (a task param always wins), plus the regime's decode-step
    count when it has one (the regime axis lives there, not in every task dict)."""
    merged = {**regime.data_knobs, **params} if regime.data_knobs else params
    if regime.new_tokens > 0:              # generation, or a load pattern derived from it
        return {**merged, "new_tokens": regime.new_tokens}
    return merged


def _stack(mats):
    """Per-unit outputs -> one tensor with the units along dim -2 (dim 0 for 1-D rows), the layout
    every cell returns for a multi-unit call. None if nothing was produced."""
    import torch

    mats = [m for m in mats if m is not None]
    if not mats:
        return None
    return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)


def _per_unit_stack(call, be, model, units):
    """Run the cell on each unit alone (its own call, no padding) and stack the outputs along the
    unit axis: the per-unit verdict of an aggregate regime, and the reference a multi-unit call is
    scored against (a padded batch is not its own valid reference: left padding shifts the position
    embeddings of an absolute-position model)."""
    return _stack([call.prepare(be, model, [unit])() for unit in units])


def _first_rows(tensor, rows):
    """The first `rows` rows of a stacked output (units run along dim -2, or dim 0 for 1-D rows);
    the whole tensor when it has fewer, so cutting never fails a measured case. A copy, so the
    saved slice carries no more storage than its rows."""
    if tensor is None:
        return None
    dim = -2 if tensor.dim() >= 2 else 0
    return tensor.narrow(dim, 0, min(rows, tensor.shape[dim])).clone()


# ---- regime drivers ---------------------------------------------------------------------------

def _timed_units(regime):
    """The units one timed call of an execution regime consumes: every unit for a multi-unit call
    (batched, or a regime whose cells consume the list as one job), else the first unit alone (an
    aggregate regime times one unit and scores every unit from its own call)."""
    if regime.kind == "batched" or not regime.aggregate:
        return list(regime.units)
    return [regime.units[0]]


def _throughput(regime, timing):
    if not timing.median_ms:
        return None
    s = timing.median_ms / 1000.0
    if regime.kind == "batched":
        return len(regime.units) / s       # units/s
    if regime.kind == "generation":
        return regime.new_tokens / s       # tokens/s (one unit's greedy decode)
    return None


@contextlib.contextmanager
def _concurrency(be, regime):
    """Whether a concurrent or open-loop regime's requests run together on this backend: a backend
    that declares `concurrent_requests` opens its session for them (the async vLLM engine moves its
    event loop to a thread every client submits to); any other backend is driven serially."""
    overlapping = regime.kind == "open" or regime.concurrency > 1
    together = overlapping and getattr(be, "concurrent_requests", False)
    if not together:
        yield False
        return
    with be.concurrent_session():
        yield True


def _run_load(regime, prepare, be):
    """Drive `prepare(units)` (a fresh zero-argument call over a unit list) under `regime`'s load
    pattern: offline issues the units in chunks of `batch_size`, concurrent and open loop one unit
    per call. Returns (LoadResult, call outputs): the last pass's outputs, in call order, of the
    calls that cover the first `score_units` units (offline: whole calls, so possibly more units);
    the caller stacks them once the timing is recorded and cuts them to the reference's rows."""
    from ..perf.load import run_concurrent, run_offline, run_open

    keep = min(regime.score_units, len(regime.units))
    if regime.kind == "offline":
        size = regime.batch_size or len(regime.units)
        chunks = [regime.units[i:i + size] for i in range(0, len(regime.units), size)]
        result, outs = run_offline(prepare, chunks, passes=regime.passes, keep=keep)
        return result, [outs[i] for i in sorted(outs)]

    def one(unit):
        return prepare([unit])

    with _concurrency(be, regime) as together:
        if regime.kind == "open":
            result, outs = run_open(one, regime.units, rate=regime.rate, passes=regime.passes,
                                    keep=keep, serial=not together)
        else:
            result, outs = run_concurrent(one, regime.units, concurrency=regime.concurrency,
                                          passes=regime.passes, keep=keep, serial=not together)
    return result, [outs[i] for i in sorted(outs)]


def _load_record(regime, result):
    """A load-pattern record's timing: `median_latency_ms` is the median time of a whole offline
    pass, or the median request latency under concurrency and open loop (from scheduled arrival);
    `throughput` is units/s either way."""
    latency = result.median_pass_ms if regime.kind == "offline" else result.percentile(50)
    return {"median_latency_ms": latency, "std_latency_ms": None, "peak_mem_mb": result.peak_mem_mb,
            "throughput": result.throughput, "load": result.summary()}


def _measure(spec, regime, be, model, prepare):
    """One measurement of `prepare(units)` calls under the regime: `warmup` untimed calls then
    `n_trials` timed calls over the regime's timed units for an execution regime, or the load
    pattern's warm-up and passes for a load regime. Returns (timing record, output): the last
    timed call's output, or a load regime's list of call outputs (`_run_load`). The record
    carries the engine workers' peak memory over the timed calls (a load regime: over its warm-up
    and passes), read through the backend's `collective_rpc` outside the timed region."""
    worker = WorkerMemory(be, model)
    if regime.kind in LOAD_KINDS:
        worker.reset()
        result, out = _run_load(regime, prepare, be)
        return {**_load_record(regime, result), **worker.record(worker.peak_mb())}, out
    timing, out = time_cell(functools.partial(prepare, _timed_units(regime)),
                            warmup=spec.warmup, n_trials=spec.n_trials, worker_memory=worker)
    return {"median_latency_ms": timing.median_ms, "std_latency_ms": timing.std_ms,
            "peak_mem_mb": timing.peak_mem_mb, "throughput": _throughput(regime, timing),
            **worker.record(timing.worker_peak_mem_mb)}, out


def _call_bytes(regime, out):
    """Bytes of tensor data in one timed call's client-side result: the last timed call of an
    execution regime, the first kept call of a load pattern."""
    if regime.kind in LOAD_KINDS:
        return returned_bytes(out[0]) if out else 0
    return returned_bytes(out)


def _ratio(timing, denominator):
    if not denominator or not denominator.get("median_latency_ms"):
        return None
    return timing["median_latency_ms"] / denominator["median_latency_ms"]


def _plain(spec, regime, be, model, record):
    """Time the backend's own request with nothing attached over the same units, in the same call
    form and timing loop as the cells: the intervention-overhead denominator. Recorded as
    unavailable when the backend has no plain request."""
    new_tokens = regime.new_tokens or 1
    record["new_tokens"] = new_tokens

    def prepare(units):
        return functools.partial(be.plain, model, units, new_tokens=new_tokens)

    try:
        timing, _ = _measure(spec, regime, be, model, prepare)
    except NotImplementedError as error:
        record.update(unavailable=str(error), error=None)
        return None
    except Exception as error:
        _failure(record, error)
        return None
    record.update(timing, error=None)
    return timing


def _execute_regime(spec, regime, iface, be, model, meta, outputs):
    """One regime: the per-unit reference stack a multi-unit call is scored against, then the
    baseline cell, the plain request and every case, each measured under the regime, then the
    effect check for the single-unit regimes. Every cell error is isolated and recorded."""
    name = regime.name

    def bound(params, record):
        call = _bind_case(spec, iface, _task_params(regime, params), record)
        return call, functools.partial(call.prepare, be, model)

    reference = None
    if regime.kind == "batched":
        reference, scored = "batched_perprompt", regime.units
    elif regime.kind == "offline" and not any(k[0] == "offline_perprompt" for k in meta):
        reference, scored = "offline_perprompt", regime.units[:regime.score_units]   # once per run
    if reference is not None:
        for task in spec.tasks:
            key = (reference, task.label)
            record = meta[key] = {}
            try:
                call, _ = bound(task.params, record)
                outputs[key] = _per_unit_stack(call, be, model, scored)
                _success(record)
            except Exception as e:
                _failure(record, e)

    base = None
    baseline = meta[("__baseline__", name)] = {}
    try:
        _, prepare = bound(spec.baseline.params, baseline)
        base, _ = _measure(spec, regime, be, model, prepare)
        baseline.update(base)
        _success(baseline)
    except Exception as e:
        _failure(baseline, e)
    plain = _plain(spec, regime, be, model, meta.setdefault(("__vanilla__", name), {}))

    for label, params, alternative in _cases(spec, iface):
        key = (name, label)
        case = meta[key] = dict(deepcopy(alternative) or {})
        try:
            call, prepare = bound(params, case)
            timing, out = _measure(spec, regime, be, model, prepare)
            case.update(timing, overhead_vs_baseline=_ratio(timing, base),
                        overhead_vs_vanilla=_ratio(timing, plain),   # kept if shaping fails below
                        returned_bytes=_call_bytes(regime, out))
            if regime.kind in LOAD_KINDS:
                case["error_stage"] = "stack"
                out = _stack(out)
            if regime.kind in ("interactive", "generation") and regime.aggregate:
                case["error_stage"] = "aggregate"
                out = _per_unit_stack(call, be, model, regime.units)
            elif regime.kind == "offline":
                per_unit = outputs.get(("offline_perprompt", (alternative or {}).get("row", label)))
                if per_unit is not None:        # else scoring reports the gap
                    out = _first_rows(out, per_unit.shape[-2 if per_unit.dim() >= 2 else 0])
            outputs[key] = out
            _success(case)
        except Exception as e:              # isolated: engine survives, later cells run
            _failure(case, e)

    # the non-vacuity guard, recorded per run (score reads it off the reference side)
    if spec.effect is not None and regime.kind in ("interactive", "generation"):
        effect = meta[("__effect__", name)] = {}
        values = {}
        for side, params in (("baseline", spec.effect.baseline_params),
                             ("perturbed", spec.effect.perturbed_params)):
            record = effect[side] = {}
            try:
                call, prepare = bound(params, record)
                values[side] = (_per_unit_stack(call, be, model, regime.units) if regime.aggregate
                                else prepare(regime.units)())
                _success(record)
            except Exception as e:
                _failure(record, e)
        effect.update(error_stage="effect", strong=False)
        if len(values) == 2:
            try:
                effect.update(compute_effect_size(
                    values["baseline"], values["perturbed"],
                    tv_floor=spec.effect.tv_floor, top1_ceiling=spec.effect.top1_ceiling))
                _success(effect)
            except Exception as e:
                _failure(effect, e)
        else:
            effect["error"] = "Effect check failed; see baseline and perturbed call records"


def _run_coordinates(spec, interface: str) -> dict:
    return spec_coordinates(spec, interface)


def alternative_label(label: str, realization: dict) -> str:
    return f"{label} [{', '.join(f'{k}={v}' for k, v in sorted(realization.items()))}]"


def _cases(spec, interface: str):
    """(label, params, alternative) per executed case: each row with the system's documented
    realization and, when the run asks for them, each alternative the cell declares."""
    alternatives = (get_realizations(spec.methodology, spec.family, interface)
                    if spec.realizations == "all" else ())
    for task in spec.tasks:
        yield task.label, task.params, None
        for realization in alternatives:
            yield (alternative_label(task.label, realization), {**task.params, **realization},
                   {"row": task.label, "alternative": realization})


def _cached_revision(repo):
    """The commit of the cached snapshot `repo` loads from, or None. transformers 5.18 stopped
    setting `config._commit_hash`, so a newer build reads the revision from the hub cache instead:
    `snapshots/<commit>/config.json` is the same commit an older build reports."""
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(repo, "config.json")
    except Exception:
        return None
    if not isinstance(path, str):
        return None
    import os

    commit = os.path.basename(os.path.dirname(path))
    return commit if len(commit) == 40 else None


def _layer_count(model):
    """The loaded model's decoder layer count from its config, or None when it does not say."""
    config = getattr(model, "config", None)
    for source in (getattr(config, "text_config", None), config):
        for name in ("num_hidden_layers", "n_layer"):
            value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def execute_run(spec, run: RunConfig, out_dir: str, run_name: str,
                release_findings: list | None = None, debug: bool = False,
                backend=None, interface=None, provenance=None) -> str:
    """Run every (regime, task) cell of `spec` on `run`'s stack; write outputs + provenance.
    Returns the outputs path. Cell errors are isolated and recorded in meta, never fatal to the
    run (the engine survives; later cells still execute)."""
    prov = resolve_provenance(run) if provenance is None else deepcopy(provenance)
    be = make_backend(run, spec) if backend is None else backend
    iface = cell_interface(run) if interface is None else interface
    prov["coordinates"] = _run_coordinates(spec, iface)
    if release_findings is not None:
        prov["release_check"] = {"clean": not release_findings, "findings": release_findings}

    outputs, meta = {}, {}
    model = None
    if worker_memory_requested():
        allow_callable_rpc()             # before the engine starts: the worker-memory RPC
    try:
        model = be.load(spec.repo)
        count = _layer_count(model) if spec.n_layers is not None else None
        if count is not None and count != spec.n_layers:
            raise ValueError(f"{spec.repo} has {count} layers; the model row says {spec.n_layers}")
        if provenance is not None:
            import hashlib
            import json
            tokenizer = model.tokenizer
            vocab = tokenizer.get_vocab()
            revision = getattr(getattr(model, "config", None), "_commit_hash", None)
            if not isinstance(revision, str):
                revision = _cached_revision(spec.repo)
            prov["model_identity"] = {
                "repo": spec.repo,
                "revision": revision if isinstance(revision, str) else None,
                "vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
                "vocab_size": len(vocab),
            }
        for regime in spec.regimes:
            _execute_regime(spec, regime, iface, be, model, meta, outputs)

        dbg_out = dbg_prov = None
        if debug:       # the debug companion (isb/debugtrace.py) never feeds the verdict
            from ..debugtrace import collect_debug, debug_prompts
            from ..profiles import PROFILES
            try:
                mprof = PROFILES[spec.family]
                dbg_out, dbg_prov = collect_debug(be, model, mprof, debug_prompts(spec),
                                                  mprof.default_residual(be.name))
            except Exception:
                traceback.print_exc()
    finally:
        if model is not None:
            be.teardown(model)

    outputs[("__meta__",)] = meta
    path = save_run(out_dir, run_name, outputs, prov)
    print(f"[execute] {run_name}: {sum(1 for k in outputs if k[0] not in ('__meta__',))} outputs "
          f"+ provenance -> {path}")
    if dbg_out:
        import os as _os
        dpath = save_run(_os.path.join(out_dir, "debug"), f"{run_name}-debug",
                         dbg_out, {**prov, **dbg_prov})
        print(f"[execute] {run_name}: debug companion -> {dpath}")
    return path

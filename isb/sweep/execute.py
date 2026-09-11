"""Execute one run (design.md §12.10): (spec × data × RunConfig) -> one run file on disk.

A run is a fully described execution in its own process. It ALWAYS writes ONE self-contained
file, <out_dir>/<run_name>.pt (isb/runfile.py), holding:

    outputs      {(regime_kind, label): cpu tensor} — every cell's warm output, plus
                 ("batched_perprompt", label): the per-prompt stack for each batched task, so ANY
                 run can later serve as the reference for a padded-batch comparison; plus
                 ("__meta__",): per-cell perf/errors and the params each case ran with.
    provenance   the four-layer record (client/deployment/engine/host) plus the run's coordinates
                 (isb/runs.py `run_coordinates`: spec, data sources, regimes, cases, protocol) and,
                 in release mode, the clean-environment findings the numbers were certified under.

Scoring is a SEPARATE, pure step (score.py) over any two such run files: execution never
compares, so results can be re-scored, cross-compared, and audited without re-running.
"""
from __future__ import annotations

import inspect
import traceback
from copy import deepcopy
from dataclasses import dataclass, field

from ..methodologies.observations import capture_choices
from ..methodologies.registry import get_cell
from ..perf.timing import time_cell
from ..runfile import save_run
from ..runs import RunConfig, cell_interface, make_backend, resolve_provenance, spec_coordinates
from ..sweep.guards import compute_effect_size


# ---- cell-call helpers (shared with tests; formerly the sweep driver's) -----------------------

def _cell(name, methodology, family):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise LookupError(f"no cell for {methodology}/{family}/{name}")
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

    def prepare(self, impl, model, prompts):
        owned = deepcopy(self.params)
        positional = [owned.pop(name) for name in self.positional_names]

        def invoke():
            with capture_choices(self.params, self.record):
                return self.fn(impl, model, prompts, *positional, **owned)

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
    traceback.print_exc()
    record["error"] = repr(error)[:1000]


def _success(record):
    record.update(error=None, error_stage=None)


def _task_params(regime, params):
    """Per-call cell params = dataset knobs (per-source defaults, e.g. the upstream readout-position
    rule) UNDER the task's explicit params (a task param always wins), plus the generation regime's
    decode-step count from the ExecutionRegime (the regime axis lives there, not in every task dict)."""
    merged = {**regime.data_knobs, **params} if regime.data_knobs else params
    if regime.kind == "generation":
        return {**merged, "new_tokens": regime.new_tokens}
    return merged


def _unit(u):
    """One trace unit -> the prompt list the cell consumes. A unit is either a prompt string (its own
    single-prompt trace) or a (clean, corrupted) tuple (activation patching consumes the pair as one
    unit). This is what lets a pair be a first-class trace in an aggregate workload, identical to a
    single prompt, so N pairs stack like N prompts."""
    return list(u) if isinstance(u, tuple) else [u]


def _per_prompt_stack(call, impl, model, prompts):
    """Run the cell on each UNIT alone (its own trace, no padding) and stack the per-unit outputs
    along the sample dim. A unit is a prompt string or a (clean, corrupted) pair (`_unit`). Two uses:
      - aggregate-interactive verdict: N independent single-unit traces so the oracle scores top-1
        agreement as a FRACTION over N + mean TV over N — robust, not a single-token anecdote.
      - batched reference: the per-prompt ground truth a padded batch is scored against (a single
        padded batch is NOT its own valid reference for absolute-position models — GPT-2 left-padding
        shifts the position embeddings on padded rows).
    Same concat rule the batched cells use (dim=-2), so [.,vocab] -> [N,vocab] and [L,.,vocab] ->
    [L,N,vocab]. Returns None if nothing was produced."""
    import torch

    mats = [
        call.prepare(impl, model, _unit(p))()
        for p in prompts
    ]
    mats = [m for m in mats if m is not None]
    if not mats:
        return None
    return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)


def _throughput(regime, timing):
    if not timing.median_ms:
        return None
    s = timing.median_ms / 1000.0
    if regime.kind == "batched":
        return len(regime.prompts) / s     # prompts/s
    if regime.kind == "generation":
        return regime.new_tokens / s       # tokens/s (single-prompt greedy decode)
    return None


def _run_coordinates(spec, interface: str) -> dict:
    return spec_coordinates(spec, interface)


def execute_run(spec, run: RunConfig, out_dir: str, run_name: str,
                release_findings: list | None = None, debug: bool = False,
                backend=None, interface=None, provenance=None) -> str:
    """Run every (execution regime, task) cell of `spec` on `run`'s stack; write outputs + provenance.
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
    try:
        model = be.load(spec.repo)
        if provenance is not None:
            import hashlib
            import json
            tokenizer = model.tokenizer
            vocab = tokenizer.get_vocab()
            revision = getattr(getattr(model, "config", None), "_commit_hash", None)
            prov["model_identity"] = {
                "repo": spec.repo,
                "revision": revision if isinstance(revision, str) else None,
                "vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest(),
                "vocab_size": len(vocab),
            }
        for regime in spec.regimes:
            timed_prompts = ([regime.prompts[0]] if regime.aggregate
                             else regime.prompts)
            if regime.aggregate and isinstance(regime.prompts[0], tuple):
                timed_prompts = list(regime.prompts[0])
            base_timing = None
            baseline = meta[("__baseline__", regime.kind)] = {}
            try:
                call = _bind_case(spec, iface, _task_params(regime, spec.baseline.params), baseline)
                base_timing, _ = time_cell(
                    lambda: call.prepare(be, model, timed_prompts),
                    warmup=spec.warmup, n_trials=spec.n_trials)
                baseline["median_latency_ms"] = base_timing.median_ms
                _success(baseline)
            except Exception as e:
                _failure(baseline, e)

            for task in spec.tasks:
                key = (regime.kind, task.label)
                case = meta[key] = {}
                try:
                    call = _bind_case(spec, iface, _task_params(regime, task.params), case)
                    timing, warm = time_cell(
                        lambda: call.prepare(be, model, timed_prompts),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                    if regime.aggregate:
                        case["error_stage"] = "aggregate"
                        warm = _per_prompt_stack(call, be, model, regime.prompts)
                    outputs[key] = warm
                    case.update({
                        "median_latency_ms": timing.median_ms, "std_latency_ms": timing.std_ms,
                        "peak_mem_mb": timing.peak_mem_mb,
                        "overhead_vs_baseline": (timing.median_ms / base_timing.median_ms)
                        if (base_timing and base_timing.median_ms) else None,
                        "throughput": _throughput(regime, timing),
                    })
                    _success(case)
                except Exception as e:              # isolated: engine survives, later cells run
                    _failure(case, e)

            if regime.kind == "batched":          # any run can reference a padded-batch compare
                for task in spec.tasks:
                    key = ("batched_perprompt", task.label)
                    record = meta[key] = {}
                    try:
                        call = _bind_case(spec, iface, _task_params(regime, task.params), record)
                        outputs[key] = _per_prompt_stack(call, be, model, regime.prompts)
                        _success(record)
                    except Exception as e:
                        _failure(record, e)

            # the non-vacuity guard, recorded per run (score reads it off the reference side)
            if spec.effect is not None and regime.kind in ("interactive", "generation"):
                effect = meta[("__effect__", regime.kind)] = {}
                values = {}
                for side, params in (("baseline", spec.effect.baseline_params),
                                     ("perturbed", spec.effect.perturbed_params)):
                    record = effect[side] = {}
                    try:
                        call = _bind_case(spec, iface, _task_params(regime, params), record)
                        values[side] = (_per_prompt_stack(call, be, model, regime.prompts)
                                        if regime.aggregate
                                        else call.prepare(be, model, regime.prompts)())
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

        dbg_out = dbg_prov = None
        if debug:       # the debug companion (isb/debugtrace.py) — never feeds the verdict
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

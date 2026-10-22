"""Execute one run (design.md §12.10): (spec × data × RunConfig) -> one run file on disk.

A run is a fully described execution in its own process. It ALWAYS writes ONE self-contained
file, <out_dir>/<run_name>.pt (isb/runfile.py), holding:

    outputs      {(regime_kind, label): cpu tensor} — every cell's warm output (a declared
                 alternative realization is its own label, `<row> [k=v]`, design.md §12.16), plus
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

import contextlib
import inspect
import traceback
from copy import deepcopy
from dataclasses import dataclass, field

from ..methodologies.observations import capture_choices
from ..methodologies.registry import Unsupported, get_cell, get_realizations
from ..perf.timing import time_cell
from ..runfile import save_run
from ..runs import RunConfig, cell_interface, make_backend, resolve_provenance, spec_coordinates
from ..sweep.guards import compute_effect_size
from ..sweep.spec import LOAD_KINDS


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
    if isinstance(error, Unsupported):     # the system lacks the capability (design.md §12.14)
        record.update(unsupported=str(error), error=None, error_stage=None)
        return
    traceback.print_exc()
    record["error"] = repr(error)[:1000]


def _success(record):
    record.update(error=None, error_stage=None)


def _task_params(regime, params):
    """Per-call cell params = dataset knobs (per-source defaults, e.g. the upstream readout-position
    rule) UNDER the task's explicit params (a task param always wins), plus the regime's decode-step
    count when it has one (the regime axis lives there, not in every task dict)."""
    merged = {**regime.data_knobs, **params} if regime.data_knobs else params
    if regime.new_tokens > 0:              # generation, or a load pattern derived from it
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


def _time_vanilla(spec, regime, be, model, timed_prompts, record):
    """Time the backend's own request with nothing attached, in the same job and regime as the
    cells: the intervention-overhead denominator (design.md §12.14). Recorded as unavailable when
    the backend has no plain request or the regime is batched."""
    if regime.kind not in ("interactive", "generation"):
        record["unavailable"] = f"no plain request for the {regime.kind} regime"
        return None
    prompt = timed_prompts[0]              # the first timed unit (a paired unit's clean prompt)
    new_tokens = regime.new_tokens if regime.kind == "generation" else 1
    record["new_tokens"] = new_tokens
    try:
        timing, _ = time_cell(lambda: (lambda: be.vanilla(model, prompt, new_tokens=new_tokens)),
                              warmup=spec.warmup, n_trials=spec.n_trials)
    except NotImplementedError as error:
        record.update(unavailable=str(error), error=None)
        return None
    except Exception as error:
        _failure(record, error)
        return None
    record.update(median_latency_ms=timing.median_ms, std_latency_ms=timing.std_ms, error=None)
    return timing


# ---- load patterns (design.md §12.17) ---------------------------------------------------------

def _stack(mats):
    """Per-unit outputs -> one tensor, the same concat rule as `_per_prompt_stack`."""
    import torch

    mats = [m for m in mats if m is not None]
    if not mats:
        return None
    return torch.cat(mats, dim=-2) if mats[0].dim() >= 2 else torch.stack(mats)


def _first_units(tensor, keep):
    """The first `keep` units of a stacked output (units run along dim -2, or dim 0 for 1-D rows)."""
    if tensor is None:
        return None
    return tensor.narrow(-2 if tensor.dim() >= 2 else 0, 0, keep)


@contextlib.contextmanager
def _concurrency(be, regime):
    """Whether a concurrent regime's requests run together on this backend: a backend that
    declares `concurrent_requests` opens its session for them (the async vLLM engine moves its
    event loop to a thread every client submits to); any other backend is driven serially."""
    together = regime.concurrency > 1 and getattr(be, "concurrent_requests", False)
    if not together:
        yield False
        return
    with be.concurrent_session():
        yield True


def _run_load(regime, prepare_units, be):
    """Drive one bound call under `regime`'s load pattern. `prepare_units(units)` returns a fresh
    zero-argument call over that unit list. Returns (LoadResult, scored output): the stacked
    output of the first `score_units` units of the last pass."""
    from ..perf.load import run_concurrent, run_offline

    keep = min(regime.score_units, len(regime.prompts))
    if regime.kind == "offline":
        size = regime.batch_size or len(regime.prompts)
        chunks = [regime.prompts[i:i + size] for i in range(0, len(regime.prompts), size)]
        result, outs = run_offline(prepare_units, chunks, passes=regime.passes, keep=keep)
        return result, _first_units(_stack([outs[i] for i in sorted(outs)]), keep)
    with _concurrency(be, regime) as together:
        result, outs = run_concurrent(lambda unit: prepare_units(_unit(unit)), regime.prompts,
                                      concurrency=regime.concurrency, passes=regime.passes,
                                      keep=keep, serial=not together)
    return result, _stack([outs[i] for i in sorted(outs)])


def _load_record(regime, result):
    """A load-pattern record's timing: `median_latency_ms` is the median time of a whole offline
    pass, or the median request latency under concurrency; `throughput` is units/s either way."""
    latency = result.median_pass_ms if regime.kind == "offline" else result.percentile(50)
    return {"median_latency_ms": latency, "std_latency_ms": None, "peak_mem_mb": result.peak_mem_mb,
            "throughput": result.throughput, "load": result.summary()}


def _time_vanilla_load(regime, be, model, record):
    """The plain system under the same load pattern: offline sends each chunk as one plain
    request (`vanilla_many`), concurrent sends each unit's plain request from the same clients."""
    first = (lambda unit: unit[0] if isinstance(unit, tuple) else unit)
    new_tokens = regime.new_tokens or 1
    record["new_tokens"] = new_tokens
    if regime.kind == "offline":
        many = getattr(be, "vanilla_many", None)
        if many is None:
            record["unavailable"] = f"{getattr(be, 'name', 'backend')} has no plain multi-prompt request"
            return None

        def prepare(units):
            return lambda: many(model, [first(u) for u in units], new_tokens=new_tokens)
    else:
        def prepare(units):
            return lambda: be.vanilla(model, first(units[0]), new_tokens=new_tokens)
    try:
        result, _ = _run_load(regime, prepare, be)
    except NotImplementedError as error:
        record.update(unavailable=str(error), error=None)
        return None
    except Exception as error:
        _failure(record, error)
        return None
    record.update(_load_record(regime, result), error=None)
    return result


def _execute_load(spec, regime, iface, r_iface, be, model, meta, outputs):
    """One load-pattern regime (design.md §12.17): the baseline, the plain system and every case
    under the same pattern, each recorded like a regime cell. An offline regime also records the
    per-unit outputs of its scored units once (`offline_perprompt`), the reference an offline
    candidate is scored against; a concurrent regime's calls are single-unit already."""
    name = regime.name

    def bound(params, record):
        call = _bind_case(spec, r_iface, _task_params(regime, params), record)
        return call, (lambda units: call.prepare(be, model, units))

    base = None
    baseline = meta[("__baseline__", name)] = {}
    try:
        _, prepare = bound(spec.baseline.params, baseline)
        base, _ = _run_load(regime, prepare, be)
        baseline.update(_load_record(regime, base))
        _success(baseline)
    except Exception as e:
        _failure(baseline, e)
    plain = _time_vanilla_load(regime, be, model, meta.setdefault(("__vanilla__", name), {}))

    for label, params, alternative in _cases(spec, iface):
        key = (name, label)
        case = meta[key] = dict(deepcopy(alternative) or {})
        try:
            _, prepare = bound(params, case)
            result, scored = _run_load(regime, prepare, be)
            outputs[key] = scored
            latency = _load_record(regime, result)
            case.update(latency)
            case.update({
                "overhead_vs_baseline": latency["median_latency_ms"] / _load_record(regime, base)["median_latency_ms"]
                if base else None,
                "overhead_vs_vanilla": latency["median_latency_ms"] / _load_record(regime, plain)["median_latency_ms"]
                if plain else None,
            })
            _success(case)
        except Exception as e:              # isolated: engine survives, later cells run
            _failure(case, e)

    if regime.kind == "offline" and not any(k[0] == "offline_perprompt" for k in meta):
        keep = min(regime.score_units, len(regime.prompts))
        for task in spec.tasks:
            key = ("offline_perprompt", task.label)
            record = meta[key] = {}
            try:
                call, _ = bound(task.params, record)
                outputs[key] = _per_prompt_stack(call, be, model, regime.prompts[:keep])
                _success(record)
            except Exception as e:
                _failure(record, e)


def _load_sync_twin(be, repo):
    """A loaded sync-mode engine for an async run's batched regime, or None.

    nnsight's async vLLM path submits one engine request per trace and refuses several
    invokes (`AsyncVLLMBackend` raises NotImplementedError; upstream pins that with
    `test_multi_invoke_raises`, so it is the intended contract). Multi-invoke batching is a
    sync-engine capability (`VLLM._collect` merges one request per invoke), so the batched
    regime runs on a sync twin sharing the async run's engine config.

    Returns None when the run's backend isn't the in-process async one, or when it carries
    parallelism the sync constructor doesn't take (TP/PP runs keep the async engine and
    record its refusal per cell). Both engines are resident together, which fits the
    coverage default (gpu_memory_utilization 0.2 each) but not a big-model fraction like
    0.9 — a failed twin load falls back the same way, recorded per cell.
    """
    from ..backends import VLLMAsyncBackend, VLLMSyncBackend

    if not isinstance(be, VLLMAsyncBackend):
        return None
    if be.tensor_parallel_size > 1 or be.pipeline_parallel_size > 1:
        return None
    twin = VLLMSyncBackend(dtype=be.dtype, trust_remote_code=be.trust_remote_code,
                           max_model_len=be.max_model_len, tokenizer=be.tokenizer,
                           enable_prefix_caching=be.enable_prefix_caching)
    return twin, twin.load(repo, gpu_memory_utilization=be.gpu_memory_utilization)


def _run_coordinates(spec, interface: str) -> dict:
    return spec_coordinates(spec, interface)


def alternative_label(label: str, realization: dict) -> str:
    return f"{label} [{', '.join(f'{k}={v}' for k, v in sorted(realization.items()))}]"


def _cases(spec, interface: str):
    """(label, params, alternative) per executed case: each row with the system's documented
    realization and, when the run asks for them, each alternative the cell declares (§12.16)."""
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
    twin = None
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
            # The batched regime of an async run executes on a sync-mode twin engine:
            # the async engine refuses several invokes per trace (see _load_sync_twin).
            # Everything else in the loop (interactive/generation cells, effect guard)
            # keeps the run's own backend.
            r_iface, r_be, r_model = iface, be, model
            if regime.kind in ("batched", "offline") and iface == "vllm_async":
                if twin is None:
                    try:
                        twin = _load_sync_twin(be, spec.repo)
                    except Exception:               # twin load failed: cells record the
                        traceback.print_exc()       # async engine's refusal instead
                if twin is not None:
                    r_iface, (r_be, r_model) = "vllm_sync", twin
                    prov["coordinates"]["batched_interface"] = r_iface

            if regime.kind in LOAD_KINDS:
                _execute_load(spec, regime, iface, r_iface, r_be, r_model, meta, outputs)
                continue

            timed_prompts = ([regime.prompts[0]] if regime.aggregate
                             else regime.prompts)
            if regime.aggregate and isinstance(regime.prompts[0], tuple):
                timed_prompts = list(regime.prompts[0])
            base_timing = None
            baseline = meta[("__baseline__", regime.kind)] = {}
            try:
                call = _bind_case(spec, r_iface, _task_params(regime, spec.baseline.params), baseline)
                base_timing, _ = time_cell(
                    lambda: call.prepare(r_be, r_model, timed_prompts),
                    warmup=spec.warmup, n_trials=spec.n_trials)
                baseline["median_latency_ms"] = base_timing.median_ms
                _success(baseline)
            except Exception as e:
                _failure(baseline, e)

            vanilla_timing = _time_vanilla(spec, regime, r_be, r_model, timed_prompts,
                                           meta.setdefault(("__vanilla__", regime.kind), {}))

            for label, params, alternative in _cases(spec, iface):
                key = (regime.kind, label)
                case = meta[key] = dict(deepcopy(alternative) or {})
                try:
                    call = _bind_case(spec, r_iface, _task_params(regime, params), case)
                    timing, warm = time_cell(
                        lambda: call.prepare(r_be, r_model, timed_prompts),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                    if regime.aggregate:
                        case["error_stage"] = "aggregate"
                        warm = _per_prompt_stack(call, r_be, r_model, regime.prompts)
                    outputs[key] = warm
                    case.update({
                        "median_latency_ms": timing.median_ms, "std_latency_ms": timing.std_ms,
                        "peak_mem_mb": timing.peak_mem_mb,
                        "overhead_vs_baseline": (timing.median_ms / base_timing.median_ms)
                        if (base_timing and base_timing.median_ms) else None,
                        "overhead_vs_vanilla": (timing.median_ms / vanilla_timing.median_ms)
                        if (vanilla_timing and vanilla_timing.median_ms) else None,
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
                        call = _bind_case(spec, r_iface, _task_params(regime, task.params), record)
                        outputs[key] = _per_prompt_stack(call, r_be, r_model, regime.prompts)
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
        if twin is not None:
            twin[0].teardown(twin[1])
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

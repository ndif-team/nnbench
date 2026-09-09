"""Execute one run (design.md §12.10): (spec × data × RunConfig) -> one run file on disk.

A run is a fully described execution in its own process. It ALWAYS writes ONE self-contained
file, <out_dir>/<run_name>.pt (isb/runfile.py), holding:

    outputs      {(regime_kind, label): cpu tensor} — every cell's warm output, plus
                 ("batched_perprompt", label): the per-prompt stack for each batched task, so ANY
                 run can later serve as the reference for a padded-batch comparison; plus
                 ("__meta__",): per-cell perf/errors.
    provenance   the four-layer record (client/deployment/engine/host) plus the run's coordinates
                 (spec, data sources, tasks) and, in release mode, the clean-environment findings
                 the numbers were certified under.

Scoring is a SEPARATE, pure step (score.py) over any two such run files: execution never
compares, so results can be re-scored, cross-compared, and audited without re-running.
"""
from __future__ import annotations

import traceback

from ..methodologies.registry import get_cell
from ..perf.timing import time_cell
from ..protocol import describe_task
from ..runfile import save_run
from ..runs import RunConfig, cell_interface, make_backend, resolve_provenance
from ..sweep.guards import compute_effect_size


# ---- cell-call helpers (shared with tests; formerly the sweep driver's) -----------------------

def _call_cell(name, impl, model, prompts, methodology, family, params):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise LookupError(f"no cell for {methodology}/{family}/{name}")
    return fn(impl, model, prompts, **params)


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


def _per_prompt_stack(name, impl, model, prompts, methodology, family, params):
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
        _call_cell(name, impl, model, _unit(p), methodology, family, params)
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


def _run_coordinates(spec, run: RunConfig) -> dict:
    def case(task, params):
        if spec.protocol is None:
            return {"label": task.label,
                    "semantics": {k: v for k, v in params.items() if k not in task.realization},
                    "realization": {k: params[k] for k in task.realization}}
        semantics, realization = spec.protocol.classify(params)
        protocol = describe_task(spec.methodology, params, template=spec.protocol,
                                 family=spec.family)
        return {"label": task.label, "semantics": semantics, "realization": realization,
                "protocol": protocol.coordinate()}

    coordinates = {
        "spec": spec.name,
        "methodology": spec.methodology,
        "family": spec.family,
        "repo": spec.repo,
        "data": sorted({w.data_name for w in spec.regimes if w.data_name}),
        "regimes": [{"kind": w.kind, "units": len(w.prompts), "data": w.data_name,
                     "new_tokens": w.new_tokens, "aggregate": w.aggregate,
                     "data_knobs": dict(w.data_knobs),
                     "cases": [case(task, _task_params(w, task.params)) for task in spec.tasks]}
                    for w in spec.regimes],
        "cases": [case(task, task.params) for task in spec.tasks],
        "interface": cell_interface(run),
    }
    if spec.protocol is not None:
        coordinates["protocol_template"] = spec.protocol.coordinate()
    return coordinates


def execute_run(spec, run: RunConfig, out_dir: str, run_name: str,
                release_findings: list | None = None, debug: bool = False,
                backend=None, interface=None, provenance=None) -> str:
    """Run every (execution regime, task) cell of `spec` on `run`'s stack; write outputs + provenance.
    Returns the outputs path. Cell errors are isolated and recorded in meta, never fatal to the
    run (the engine survives; later cells still execute)."""
    prov = resolve_provenance(run) if provenance is None else dict(provenance)
    prov["coordinates"] = _run_coordinates(spec, run)
    if release_findings is not None:
        prov["release_check"] = {"clean": not release_findings, "findings": release_findings}

    be = make_backend(run, spec) if backend is None else backend
    iface = cell_interface(run) if interface is None else interface
    prov["coordinates"]["interface"] = iface
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
            try:
                base_timing, _ = time_cell(
                    lambda tp=timed_prompts: _call_cell(
                        iface, be, model, tp, spec.methodology, spec.family,
                        _task_params(regime, spec.baseline.params)),
                    warmup=spec.warmup, n_trials=spec.n_trials)
            except Exception:
                traceback.print_exc()

            for task in spec.tasks:
                params, label = task.params, task.label
                key = (regime.kind, label)
                try:
                    timing, warm = time_cell(
                        lambda tp=timed_prompts, p=_task_params(regime, params): _call_cell(
                            iface, be, model, tp, spec.methodology, spec.family, p),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                    if regime.aggregate:
                        warm = _per_prompt_stack(iface, be, model, regime.prompts,
                                                 spec.methodology, spec.family,
                                                 _task_params(regime, params))
                    outputs[key] = warm
                    meta[key] = {
                        "median_latency_ms": timing.median_ms, "std_latency_ms": timing.std_ms,
                        "peak_mem_mb": timing.peak_mem_mb,
                        "overhead_vs_baseline": (timing.median_ms / base_timing.median_ms)
                        if (base_timing and base_timing.median_ms) else None,
                        "throughput": _throughput(regime, timing), "error": None,
                    }
                except Exception as e:              # isolated: engine survives, later cells run
                    traceback.print_exc()
                    meta[key] = {"error": repr(e)[:300]}

            if regime.kind == "batched":          # any run can reference a padded-batch compare
                for task in spec.tasks:
                    params, label = task.params, task.label
                    try:
                        outputs[("batched_perprompt", label)] = _per_prompt_stack(
                            iface, be, model, regime.prompts,
                            spec.methodology, spec.family, _task_params(regime, params))
                    except Exception:
                        traceback.print_exc()

            # the non-vacuity guard, recorded per run (score reads it off the reference side)
            if spec.effect is not None and regime.kind in ("interactive", "generation"):
                try:
                    pairs = regime.prompts and isinstance(regime.prompts[0], tuple)
                    call = (_per_prompt_stack if regime.aggregate and pairs else _call_cell)
                    b = call(iface, be, model, regime.prompts, spec.methodology,
                             spec.family, _task_params(regime, spec.effect.baseline_params))
                    p = call(iface, be, model, regime.prompts, spec.methodology,
                             spec.family, _task_params(regime, spec.effect.perturbed_params))
                    meta[("__effect__", regime.kind)] = compute_effect_size(
                        b, p, tv_floor=spec.effect.tv_floor, top1_ceiling=spec.effect.top1_ceiling)
                except Exception:
                    traceback.print_exc()

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

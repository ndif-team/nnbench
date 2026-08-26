"""Execute one run (design.md §12.10): (spec × data × RunConfig) -> one run file on disk.

A run is a fully described execution in its own process. It ALWAYS writes ONE self-contained
file, <out_dir>/<run_name>.pt (isb/runfile.py), holding:

    outputs      {(workload_kind, label): cpu tensor} — every cell's warm output, plus
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
from ..runfile import save_run
from ..runs import RunConfig, cell_interface, make_backend, resolve_provenance
from ..sweep.guards import compute_effect_size


# ---- cell-call helpers (shared with tests; formerly the sweep driver's) -----------------------

def _call_cell(name, impl, model, prompts, methodology, family, params):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise LookupError(f"no cell for {methodology}/{family}/{name}")
    return fn(impl, model, prompts, **params)


def _task_params(workload, params):
    """Per-call cell params = dataset knobs (per-source defaults, e.g. the upstream readout-position
    rule) UNDER the task's explicit params (a task param always wins), plus the generation regime's
    decode-step count from the Workload (the regime axis lives there, not in every task dict)."""
    merged = {**workload.data_knobs, **params} if workload.data_knobs else params
    if workload.kind == "generation":
        return {**merged, "new_tokens": workload.new_tokens}
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


def _throughput(workload, timing):
    if not timing.median_ms:
        return None
    s = timing.median_ms / 1000.0
    if workload.kind == "batched":
        return len(workload.prompts) / s     # prompts/s
    if workload.kind == "generation":
        return workload.new_tokens / s       # tokens/s (single-prompt greedy decode)
    return None


def _run_coordinates(spec, run: RunConfig) -> dict:
    return {
        "spec": spec.name,
        "methodology": spec.methodology,
        "family": spec.family,
        "repo": spec.repo,
        "data": sorted({w.data_name for w in spec.workloads if w.data_name}),
        "workloads": [{"kind": w.kind, "units": len(w.prompts), "data": w.data_name}
                      for w in spec.workloads],
        "tasks": [label for _, label in spec.tasks],
        "interface": cell_interface(run),
    }


def execute_run(spec, run: RunConfig, out_dir: str, run_name: str,
                release_findings: list | None = None, debug: bool = False) -> str:
    """Run every (workload, task) cell of `spec` on `run`'s stack; write outputs + provenance.
    Returns the outputs path. Cell errors are isolated and recorded in meta, never fatal to the
    run (the engine survives; later cells still execute)."""
    import torch

    prov = resolve_provenance(run)
    prov["coordinates"] = _run_coordinates(spec, run)
    if release_findings is not None:
        prov["release_check"] = {"clean": not release_findings, "findings": release_findings}

    be = make_backend(run, spec)
    iface = cell_interface(run)
    outputs, meta = {}, {}
    model = None
    try:
        model = be.load(spec.repo)
        for workload in spec.workloads:
            timed_prompts = ([workload.prompts[0]] if workload.aggregate
                             else workload.prompts)
            if workload.aggregate and isinstance(workload.prompts[0], tuple):
                timed_prompts = list(workload.prompts[0])
            base_timing = None
            try:
                base_timing, _ = time_cell(
                    lambda tp=timed_prompts: _call_cell(
                        iface, be, model, tp, spec.methodology, spec.family,
                        _task_params(workload, spec.baseline.params)),
                    warmup=spec.warmup, n_trials=spec.n_trials)
            except Exception:
                traceback.print_exc()

            for params, label in spec.tasks:
                key = (workload.kind, label)
                try:
                    timing, warm = time_cell(
                        lambda tp=timed_prompts, p=_task_params(workload, params): _call_cell(
                            iface, be, model, tp, spec.methodology, spec.family, p),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                    if workload.aggregate:
                        warm = _per_prompt_stack(iface, be, model, workload.prompts,
                                                 spec.methodology, spec.family,
                                                 _task_params(workload, params))
                    outputs[key] = warm
                    meta[key] = {
                        "median_latency_ms": timing.median_ms, "std_latency_ms": timing.std_ms,
                        "peak_mem_mb": timing.peak_mem_mb,
                        "overhead_vs_baseline": (timing.median_ms / base_timing.median_ms)
                        if (base_timing and base_timing.median_ms) else None,
                        "throughput": _throughput(workload, timing), "error": None,
                    }
                except Exception as e:              # isolated: engine survives, later cells run
                    traceback.print_exc()
                    meta[key] = {"error": repr(e)[:300]}

            if workload.kind == "batched":          # any run can reference a padded-batch compare
                for params, label in spec.tasks:
                    try:
                        outputs[("batched_perprompt", label)] = _per_prompt_stack(
                            iface, be, model, workload.prompts,
                            spec.methodology, spec.family, _task_params(workload, params))
                    except Exception:
                        traceback.print_exc()

            # the non-vacuity guard, recorded per run (score reads it off the reference side)
            if spec.effect is not None and workload.kind in ("interactive", "generation"):
                try:
                    pairs = workload.prompts and isinstance(workload.prompts[0], tuple)
                    call = (_per_prompt_stack if workload.aggregate and pairs else _call_cell)
                    b = call(iface, be, model, workload.prompts, spec.methodology,
                             spec.family, _task_params(workload, spec.effect.baseline_params))
                    p = call(iface, be, model, workload.prompts, spec.methodology,
                             spec.family, _task_params(workload, spec.effect.perturbed_params))
                    meta[("__effect__", workload.kind)] = compute_effect_size(
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

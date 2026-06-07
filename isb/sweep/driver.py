"""The one sweep driver (design.md §12) — replaces the 5 `scripts/smoke_*.py` main loops.

Single pass, amortized load: for each `(family, backend)` load the model ONCE (intervention errors
are isolated by nnsight's deferred-exception mechanism, so a failing cell does not poison later
cells — no per-cell reload needed), then for each `(workload, task)` run the cell **warm** with
`time_cell`, capturing one warm output for the oracle AND timing the trials. Correctness is therefore
checked in the exact regime (warm, and — once wired — batched) where perf is measured. After both
backends run, the oracle scores each `(workload, task)` group HF-vs-vLLM and the report prints the
applicability map + the performance table.
"""
from __future__ import annotations

import traceback

from ..backends import HFBackend, VLLMAsyncBackend
from ..methodologies.registry import get_cell
from ..perf.timing import time_cell
from ..report import print_map, print_perf
from ..runner.run import (
    CellResult,
    PerfResult,
    disambiguate_precision,
    evaluate,
    run_cell,
)
from ..states import AppState
from .guards import compute_effect_size


def _default_backend(name, spec):
    if name == "hf":
        return HFBackend(**spec.hf_kwargs)
    if name == "vllm_async":
        return VLLMAsyncBackend(**spec.vllm_kwargs)
    raise ValueError(f"unknown backend {name!r}")


def _call_cell(name, impl, model, prompts, methodology, family, params):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise LookupError(f"no cell for {methodology}/{family}/{name}")
    return fn(impl, model, prompts, **params)


def _throughput(workload, timing):
    if workload.kind == "batched" and timing.median_ms:
        return len(workload.prompts) / (timing.median_ms / 1000.0)   # prompts/s
    return None


def _fp32_rerun(spec, params, workload):
    """Re-run the failing vLLM cell at the control's precision (a fresh fp32 engine) to separate a
    precision degradation from a real bug — fed to disambiguate_precision."""
    def rerun(c):
        return run_cell(
            spec.methodology, spec.family, c.backend,
            VLLMAsyncBackend(dtype=spec.dtype_control),
            spec.repo, workload.prompts, params=params, label="",
        ).value
    return rerun


def run_sweep(spec, backends=("hf", "vllm_async"), backend_factory=None):
    backend_factory = backend_factory or _default_backend
    results = []
    effect = {}   # spec-level effect-size verdict, computed on the HF control (interactive)

    for name in backends:
        impl = backend_factory(name, spec)
        model = None
        try:
            model = impl.load(spec.repo)
            for workload in spec.workloads:
                # no-intervention baseline: warms the engine AND is the overhead denominator
                base_timing = None
                try:
                    base_timing, _ = time_cell(
                        lambda w=workload: _call_cell(name, impl, model, w.prompts,
                                                      spec.methodology, spec.family, spec.baseline.params),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                except Exception:
                    traceback.print_exc()

                for params, label in spec.tasks:
                    try:
                        timing, warm = time_cell(
                            lambda w=workload, p=params: _call_cell(name, impl, model, w.prompts,
                                                                    spec.methodology, spec.family, p),
                            warmup=spec.warmup, n_trials=spec.n_trials)
                        perf = PerfResult(
                            median_latency_ms=timing.median_ms, std_latency_ms=timing.std_ms,
                            min_latency_ms=timing.min_ms, n_trials=timing.n_trials, warmup=timing.warmup,
                            peak_mem_mb=timing.peak_mem_mb,
                            throughput=_throughput(workload, timing),
                            overhead_vs_baseline=(timing.median_ms / base_timing.median_ms)
                            if (base_timing and base_timing.median_ms) else None,
                            enforce_eager=True if name == "vllm_async" else None)
                        results.append(CellResult(
                            spec.methodology, spec.family, name, label, "RAN",
                            latency_s=timing.median_ms / 1000.0, value=warm,
                            workload=workload.kind, perf=perf))
                    except Exception as e:                       # isolated — engine survives, continue
                        traceback.print_exc()
                        results.append(CellResult(
                            spec.methodology, spec.family, name, label, AppState.ERROR,
                            error=repr(e)[:300], workload=workload.kind))

                # effect-size guard on the control (HF, interactive), reusing the loaded model
                if name == "hf" and spec.effect is not None and workload.kind == "interactive":
                    try:
                        b = _call_cell(name, impl, model, workload.prompts, spec.methodology,
                                       spec.family, spec.effect.baseline_params)
                        p = _call_cell(name, impl, model, workload.prompts, spec.methodology,
                                       spec.family, spec.effect.perturbed_params)
                        effect["interactive"] = compute_effect_size(
                            b, p, tv_floor=spec.effect.tv_floor, top1_ceiling=spec.effect.top1_ceiling)
                    except Exception:
                        traceback.print_exc()
        finally:
            if model is not None:
                impl.teardown(model)

    # ---- oracle + report, per (workload, task) ----
    for workload in spec.workloads:
        for params, label in spec.tasks:
            cells = [c for c in results if c.workload == workload.kind and c.label == label]
            if not cells:
                continue
            hf_val = next((c.value for c in cells if c.backend == "hf"), None)
            evaluate(cells, control="hf")
            disambiguate_precision(cells, hf_val, _fp32_rerun(spec, params, workload))
            if workload.kind in effect:
                e = effect[workload.kind]
                verdict = "OK — intervention moves the control" if e["strong"] \
                    else "WEAK — verdict may be vacuous"
                print(f"\n[effect-size | {label}] control top1={e['top1_agree']:.2f} "
                      f"tv={e['tv']:.3f} -> {verdict}")
            tag = f"{label} [{workload.kind}]"
            print_map(spec.methodology, spec.family, tag, spec.repo, cells)
            print_perf(spec.methodology, spec.family, tag, spec.repo, cells)
    return results

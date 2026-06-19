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

import os
import traceback

from ..backends import HFBackend, VLLMAsyncBackend, VLLMServeBackend, VLLMSyncBackend
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


def _default_backend(name, spec, serve_host=None):
    if name == "hf":
        return HFBackend(**spec.hf_kwargs)
    if name == "vllm_async":
        return VLLMAsyncBackend(**spec.vllm_kwargs)
    if name == "vllm_sync":
        return VLLMSyncBackend(**spec.vllm_kwargs)
    if name == "vllm_serve":
        if serve_host is None:
            raise ValueError("the vllm_serve backend needs a server URL (pass serve_host= / --serve)")
        return VLLMServeBackend(host=serve_host, **spec.vllm_kwargs)
    raise ValueError(f"unknown backend {name!r}")


def _call_cell(name, impl, model, prompts, methodology, family, params):
    fn = get_cell(methodology, family, name)
    if fn is None:
        raise LookupError(f"no cell for {methodology}/{family}/{name}")
    return fn(impl, model, prompts, **params)


def _task_params(workload, params):
    """A generation workload carries its decode-step count on the Workload (it is the regime axis,
    same value for every task), so the driver injects it into the cell's params here instead of
    every task dict duplicating it (and risking divergence from the workload)."""
    if workload.kind == "generation":
        return {**params, "new_tokens": workload.new_tokens}
    return params


def _per_prompt_stack(name, impl, model, prompts, methodology, family, params):
    """Run the cell on each prompt ALONE (a 1-element list -> its own trace, no padding) and stack
    the per-prompt outputs along the sample dim. Two uses:
      - aggregate-interactive verdict: N independent single-prompt traces so the oracle scores top-1
        agreement as a FRACTION over N + mean TV over N — robust, not a single-token anecdote.
      - batched reference: the per-prompt ground truth a padded batch is scored against (a single
        padded batch is NOT its own valid reference for absolute-position models — GPT-2 left-padding
        shifts the position embeddings on padded rows).
    Same concat rule the batched cells use (dim=-2), so [.,vocab] -> [N,vocab] and [L,.,vocab] ->
    [L,N,vocab]. Returns None if nothing was produced."""
    import torch

    mats = [
        _call_cell(name, impl, model, [p], methodology, family, params)
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


def _fp32_rerun(spec, params, workload):
    """Re-run the failing vLLM cell at the control's precision (a fresh fp32 engine) to separate a
    precision degradation from a real bug — fed to disambiguate_precision. MUST match the cell's
    shape: an aggregate-interactive cell is per-prompt-stacked, so re-run it the same way (passing the
    whole prompt list would hit vLLM's batched gate and fail, making every bf16 near-tie look like a
    bug)."""
    def rerun(c):
        # match the cell's backend mode so the fp32 control is measured on the same engine, and carry
        # the spec's vLLM engine config (e.g. trust_remote_code for NemotronH) so the rerun can load
        # the same model the cell did — otherwise disambiguation silently fails on those specs.
        BE = VLLMSyncBackend if c.backend == "vllm_sync" else VLLMAsyncBackend
        impl = BE(dtype=spec.dtype_control, **spec.vllm_kwargs)
        model = None
        try:
            model = impl.load(spec.repo)
            p = _task_params(workload, params)
            if workload.aggregate:
                return _per_prompt_stack(c.backend, impl, model, workload.prompts,
                                         spec.methodology, spec.family, p)
            return _call_cell(c.backend, impl, model, workload.prompts,
                              spec.methodology, spec.family, p)
        except Exception:
            traceback.print_exc()
            return None
        finally:
            if model is not None:
                impl.teardown(model)
    return rerun


def _ref_file(dirpath, spec_name):
    return os.path.join(dirpath, f"{spec_name}.pt")


def _load_refs(refs_dir, spec_name):
    """Load the cached integrated references for VM-mode scoring: a dict {(workload_kind, label):
    cpu tensor} produced by a prior integrated run with `dump_refs=`. Returns {} if absent."""
    import torch

    path = _ref_file(refs_dir, spec_name)
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def _ctl_ref_file(dirpath, spec_name):
    return os.path.join(dirpath, f"{spec_name}__ctl.pt")


def _load_ctl_refs(refs_dir, spec_name):
    """Load cached fp32 (control-dtype) IN-PROCESS-vLLM outputs per (workload_kind, label). The
    GPU-less serve client uses these to tell a bf16 SILENTLY_WRONG that is merely a precision near-tie
    (the fp32-vLLM result matches the HF control) from a genuine mechanism bug — without a local GPU."""
    import torch

    path = _ctl_ref_file(refs_dir, spec_name)
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def dump_control_refs(spec, dirpath):
    """Cache each cell's fp32 (control-dtype) in-process-vLLM output, so a GPU-less serve run can
    disambiguate precision. This is the precomputed form of disambiguate_precision's live fp32 rerun:
    a cell that is SILENTLY_WRONG at bf16 but whose fp32-vLLM output matches the HF control is a
    precision degradation (SUPPORTED_DEGRADED), not a bug. Cells vLLM can't run at all (attention
    .source / autograd) raise and are simply skipped — they have no working version to disambiguate."""
    import torch

    impl = VLLMAsyncBackend(dtype=spec.dtype_control)
    dump, model = {}, None
    try:
        model = impl.load(spec.repo)
        for workload in spec.workloads:
            for params, label in spec.tasks:
                try:
                    # match the serve verdict's shape: aggregate-interactive is per-prompt-stacked
                    p = _task_params(workload, params)
                    if workload.aggregate:
                        val = _per_prompt_stack("vllm_async", impl, model, workload.prompts,
                                                spec.methodology, spec.family, p)
                    else:
                        val = _call_cell("vllm_async", impl, model, workload.prompts,
                                         spec.methodology, spec.family, p)
                    dump[(workload.kind, label)] = val
                except Exception:
                    traceback.print_exc()
    finally:
        if model is not None:
            impl.teardown(model)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(dump, _ctl_ref_file(dirpath, spec.name))
    print(f"[dump-ctl-refs] wrote {len(dump)} control-dtype refs -> {_ctl_ref_file(dirpath, spec.name)}")


def expected_state(spec, backend, workload_kind, label, control="hf"):
    """The cell's DECLARED expectation (the benchmark's encoded knowledge); default SUPPORTED, so only
    the known non-SUPPORTED cells need listing in `spec.expected`. Any vLLM *variant* (`vllm_serve`,
    `vllm_pp`) with no entry inherits the `vllm_async` expectation — a variant should match in-process
    single-GPU vLLM, so the interesting event is a DIVERGENCE from it (a real transport / parallelism
    surprise), not the variant reproducing vLLM's own known limitation. So a cell that ERRORs on both
    vllm_async and vllm_pp is 'expected', while vllm_pp diverging from a SUPPORTED vllm_async is the
    surprise the PP run exists to catch.

    In parallelism-equivalence mode (control != "hf") SUPPORTED_DEGRADED is not a meaningful state:
    it only ever meant "vLLM-bf16 vs HF-fp32 precision near-tie", and that comparison is gone. Both
    sides here are same-dtype vLLM, so a correct cell is simply SUPPORTED. Any declared (or inherited)
    SUPPORTED_DEGRADED expectation is therefore coerced to SUPPORTED for this mode."""
    e = spec.expected
    v = e.get((backend, workload_kind, label))
    if v is None and backend.startswith("vllm_") and backend != "vllm_async":
        v = e.get(("vllm_async", workload_kind, label))
    if control != "hf" and v == AppState.SUPPORTED_DEGRADED:
        v = AppState.SUPPORTED
    return v if v is not None else AppState.SUPPORTED


def run_sweep(spec, backends=("hf", "vllm_async"), backend_factory=None,
              serve_host=None, dump_refs=None, refs=None, ctl_refs=None,
              control="hf"):
    """Run a spec across `backends`.

    `control` names the backend the oracle scores every other cell against (default "hf", the
    cross-backend correctness check). For the PARALLELISM-EQUIVALENCE mode (`bench.py --pp/--tp`)
    it is "vllm_async": the single-GPU (1,1) vLLM run is the ground truth and the (tp,pp) candidate
    ("vllm_pp") is scored against it — "same intervention at tp=1,pp=1" (GT2). The effect-size guard
    runs on that control, confirming the intervention is non-vacuous (GT1). No HF reference involved;
    precision disambiguation (an HF-fp32 vs vLLM-bf16 concept) is skipped when control != "hf".

    VM-style serve mode: `backends=("vllm_serve",)` + `serve_host=<url>` runs the cells against a
    remote nnsight-vllm-serve server. The GPU-less client can't run the HF control, so pass
    `refs=<dir>` to score serve cells against cached integrated references (produced by a prior
    integrated run with `dump_refs=<dir>`). `dump_refs=<dir>` persists this run's HF reference (the
    interactive control output + the per-prompt-interactive batched reference) for that later VM run.
    `ctl_refs=<dir>` supplies cached fp32-vLLM outputs (from `dump_control_refs`) so the serve run can
    disambiguate a bf16 precision near-tie (SUPPORTED_DEGRADED) from a real bug without a local GPU.
    """
    backend_factory = backend_factory or (lambda name, sp: _default_backend(name, sp, serve_host))
    results = []
    effect = {}        # spec-level effect-size verdict, computed on the HF control (interactive)
    batched_refs = {}  # label -> per-prompt-interactive HF reference for the batched oracle
    loaded_refs = _load_refs(refs, spec.name) if refs else {}  # (kind,label) -> cpu tensor (VM mode)
    loaded_ctl_refs = _load_ctl_refs(ctl_refs, spec.name) if ctl_refs else {}  # fp32-vLLM, serve disambig
    ref_dump = {}      # (kind,label) -> cpu tensor to persist when dump_refs is set

    for name in backends:
        impl = backend_factory(name, spec)
        model = None
        try:
            model = impl.load(spec.repo)
            for workload in spec.workloads:
                # An aggregate-interactive workload runs each prompt as its OWN trace; perf is still the
                # single-prompt latency (design: interactive = 1 trace, 1 prompt), so TIME on prompt[0]
                # while the VERDICT below aggregates over the whole set.
                timed_prompts = (
                    [workload.prompts[0]] if workload.aggregate else workload.prompts
                )
                # no-intervention baseline: warms the engine AND is the overhead denominator
                base_timing = None
                try:
                    base_timing, _ = time_cell(
                        lambda tp=timed_prompts: _call_cell(
                            name, impl, model, tp, spec.methodology, spec.family,
                            _task_params(workload, spec.baseline.params)),
                        warmup=spec.warmup, n_trials=spec.n_trials)
                except Exception:
                    traceback.print_exc()

                for params, label in spec.tasks:
                    try:
                        timing, warm = time_cell(
                            lambda tp=timed_prompts, p=_task_params(workload, params): _call_cell(
                                name, impl, model, tp, spec.methodology, spec.family, p),
                            warmup=spec.warmup, n_trials=spec.n_trials)
                        # aggregate-interactive verdict: re-run the cell per-prompt over the full set so
                        # the oracle scores top-1 agreement as a fraction over N (the timed warm output
                        # above is a single prompt — fine for perf, too thin for a verdict).
                        if workload.aggregate:
                            warm = _per_prompt_stack(name, impl, model, workload.prompts,
                                                     spec.methodology, spec.family,
                                                     _task_params(workload, params))
                        perf = PerfResult(
                            median_latency_ms=timing.median_ms, std_latency_ms=timing.std_ms,
                            min_latency_ms=timing.min_ms, n_trials=timing.n_trials, warmup=timing.warmup,
                            peak_mem_mb=timing.peak_mem_mb,
                            throughput=_throughput(workload, timing),
                            overhead_vs_baseline=(timing.median_ms / base_timing.median_ms)
                            if (base_timing and base_timing.median_ms) else None,
                            enforce_eager=True if name in ("vllm_async", "vllm_sync") else None)
                        results.append(CellResult(
                            spec.methodology, spec.family, name, label, "RAN",
                            latency_s=timing.median_ms / 1000.0, value=warm,
                            workload=workload.kind, perf=perf))
                        # persist the interactive HF control output as the cached VM-mode reference
                        if dump_refs and name == control and workload.kind == "interactive":
                            ref_dump[("interactive", label)] = warm
                    except Exception as e:                       # isolated — engine survives, continue
                        traceback.print_exc()
                        results.append(CellResult(
                            spec.methodology, spec.family, name, label, AppState.ERROR,
                            error=repr(e)[:300], workload=workload.kind))

                # effect-size guard on the control (HF, interactive/generation — the per-trace
                # regimes a write cell runs in), reusing the loaded model
                if name == control and spec.effect is not None \
                        and workload.kind in ("interactive", "generation"):
                    try:
                        b = _call_cell(name, impl, model, workload.prompts, spec.methodology,
                                       spec.family, _task_params(workload, spec.effect.baseline_params))
                        p = _call_cell(name, impl, model, workload.prompts, spec.methodology,
                                       spec.family, _task_params(workload, spec.effect.perturbed_params))
                        effect[workload.kind] = compute_effect_size(
                            b, p, tv_floor=spec.effect.tv_floor, top1_ceiling=spec.effect.top1_ceiling)
                    except Exception:
                        traceback.print_exc()

                # per-prompt-interactive reference for the batched oracle, built on HF (the control)
                # while its model is loaded — the same padded HF batch cannot be its own reference.
                if name == control and workload.kind == "batched":
                    for params, label in spec.tasks:
                        try:
                            batched_refs[label] = _per_prompt_stack(
                                name, impl, model, workload.prompts, spec.methodology, spec.family, params)
                            if dump_refs:
                                ref_dump[("batched", label)] = batched_refs[label]
                        except Exception:
                            traceback.print_exc()
        finally:
            if model is not None:
                impl.teardown(model)

    # ---- oracle + report, per (workload, task) ----
    n_total = n_surprise = 0
    for workload in spec.workloads:
        for params, label in spec.tasks:
            cells = [c for c in results if c.workload == workload.kind and c.label == label]
            if not cells:
                continue
            # VM mode (refs given): score every cell against the cached integrated reference, since the
            # GPU-less serve client has no in-run HF control. Otherwise batched uses the per-prompt-
            # interactive reference (HF-batched is not self-consistent for absolute-position models) and
            # interactive keeps HF as the same-run control.
            if refs:
                ref = loaded_refs.get((workload.kind, label))
            else:
                ref = batched_refs.get(label) if workload.kind == "batched" else None
            hf_val = ref if ref is not None else next((c.value for c in cells if c.backend == control), None)
            evaluate(cells, control=control, ref_override=ref)
            # Precision disambiguation. In-process: re-run the flagged cell on a live fp32 engine. On
            # the GPU-less serve client that isn't possible, so use the cached fp32-vLLM control output
            # (dump_control_refs) when supplied — same logic, precomputed. Without it, a serve bf16
            # near-tie stays SILENTLY_WRONG (honest: undisambiguated, not silently passed).
            if control == "hf" and serve_host is None:
                disambiguate_precision(cells, hf_val, _fp32_rerun(spec, params, workload))
            elif control == "hf" and loaded_ctl_refs:
                disambiguate_precision(
                    cells, hf_val, lambda c: loaded_ctl_refs.get((c.workload, c.label)))
            # the benchmark's DELTA: actual vs the declared expectation (NO_REFERENCE can't be judged)
            for c in cells:
                c.expected = expected_state(spec, c.backend, c.workload, c.label, control)
                c.surprise = c.state != AppState.NO_REFERENCE and c.state != c.expected
                n_total += 1
                n_surprise += int(c.surprise)
            if workload.kind in effect:
                e = effect[workload.kind]
                verdict = "OK — intervention moves the control" if e["strong"] \
                    else "WEAK — verdict may be vacuous"
                print(f"\n[effect-size | {label}] control top1={e['top1_agree']:.2f} "
                      f"tv={e['tv']:.3f} -> {verdict}")
            tag = f"{label} [{workload.kind}]"
            print_map(spec.methodology, spec.family, tag, spec.repo, cells)
            print_perf(spec.methodology, spec.family, tag, spec.repo, cells)

    # the line a maintainer actually reads: did anything change vs what we know?
    if n_surprise:
        print(f"\n[expected] {spec.name}: {n_total - n_surprise}/{n_total} cells as expected — "
              f"⚠ {n_surprise} SURPRISE(S) (see ⚠ rows above)")
    else:
        print(f"\n[expected] {spec.name}: {n_total}/{n_total} cells as expected — no surprises")

    if dump_refs and ref_dump:
        import torch

        os.makedirs(dump_refs, exist_ok=True)
        torch.save(ref_dump, _ref_file(dump_refs, spec.name))
        print(f"\n[dump-refs] wrote {len(ref_dump)} references -> {_ref_file(dump_refs, spec.name)}")
    return results

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nnbench: a **systems performance + coverage benchmark** for interpretability workloads run through nnsight across serving backends (HuggingFace vs vLLM). It measures whether a workload **runs**, **runs correctly** (numerical equivalence vs an HF reference), and **runs fast** — it is *not* a faithfulness benchmark. The headline deliverable is the applicability map, and the dangerous state it exists to catch is `SILENTLY_WRONG`: runs with no error but produces wrong numbers (e.g. the portable logit-lens on vLLM-Llama drops half the dual residual stream).

## Environment & commands

The main benchmark discovers independent `backends/NAME/compose.yml` configurations. Every one
defines a `runner` service; its Dockerfile and entrypoint own dependencies and execution settings.
The host runner in `isb/jobs/` is backend-agnostic: do not add engine switches, backend registries,
Conda paths, or shared-Compose override selection. See `backends/README.md` for the file contract.

```bash
# Unit tests — all no-GPU (fake backends, torch-only logic)
python -m pytest tests/ -q
python -m pytest tests/test_sweep.py -q

# Benchmark (needs GPU): build images, then the host script launches one container per run
python scripts/bench.py build nnsight-hf nnsight-vllm
python scripts/bench.py run --spec steering_gpt2 --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0
python scripts/bench.py run --spec all --backends nnsight-hf --gpu 0
# Specs live in isb/specs/ (logit_lens_gpt2, logit_lens_llama, steering_gpt2, activation_patching_gpt2,
# ablation_gpt2, ...).

# Client/server (vllm_serve) split — after stocking HF and control run files
GPU=0 docker/run_vm.sh
```

The standalone micro/perf entrypoints have not yet been moved to the Compose launcher; that is a
separate follow-up to this main benchmark conversion.

vLLM EngineCore uses spawn: every entrypoint must run under an `if __name__ == "__main__"` guard.

## Architecture

The core unit is a **cell**: one explicit function per `(methodology, family, backend)`, registered with `@cell(...)` in `isb/methodologies/` (registry in `isb/methodologies/registry.py`). Variances (prompts, layers, idiomatic-vs-portable formulation) are runtime **params** to the cell, not separate registrations. This flatness is deliberate — an earlier "Resolver" abstraction that *generated* intervention code from declarations was killed (design.md §11–12); the leveled primitive model in design.md §3 only *indexes* cells, never constructs them. Do not reintroduce a construction layer. Every `vllm_*` variant cell (`vllm_serve`, `vllm_sync`, `vllm_pp`, …) falls back to the `vllm_async` cell automatically (same intervention code; the variant difference — over-HTTP, in-process-sync, pipeline/tensor-parallel — lives entirely in the backend object).

Main data flow: `scripts/bench.py` → frozen experiment/inputs → `backends/NAME/run.py` in its
own Compose project → validated artifacts → `isb/jobs/score.py`. The shared nnsight worker reuses
the scientific execution routines below. Legacy standalone execute/score tools retain their
older `.pt` interface; their orchestration conventions do not define the new runner. The manager
reads saved job reports; legacy `.pt` browsing requires explicit `--import-legacy` summaries.

1. **Spec** (`isb/specs/`): a `CellConfig` + `Workload`s (interactive / batched) per methodology. Batching is a *coverage axis* — each workload is oracle-checked in its own regime, not just timed.
2. **Backend** (`isb/backends/`): `be` objects — `hf` (the per-family control), `vllm_async` (in-process system under test), `vllm_serve` (over-HTTP). One model load per backend, amortized across all tasks; an intervention error is isolated (engine survives, later tasks still run).
3. **Oracle** (`isb/oracle/equivalence.py`): compares outputs via top-1 agreement and softmax TV. The main runner takes an explicit reference and comparison axis, with no implicit precision-control jobs. The legacy `scripts/score.py --ctl` can still disambiguate precision using old-format artifacts.
4. **Perf** (`isb/perf/`): warm timing (warmup + N trials, CUDA-synced, median±std, peak mem) — correctness is verified in the same warm/batched regime perf is measured in.
5. **Report** (`isb/report/`): applicability map (`AppState` in `isb/states.py`) + performance table.

The **micro tier** (`isb/micro/`) probes individual nnsight primitives per backend. A probe that times out is recorded `HANG` and **aborts that backend's remaining probes** — the stuck thread still owns the engine loop, so anything after it would measure a poisoned engine. Probe registration order is safest-first for this reason.

## Conventions & constraints

- **The trace body must live in the same frame as `with model.trace(...)`** — this nnsight dev branch compiles the captured body, so splitting it across a generator/`@contextmanager` yields an empty body and deadlocks. `be.run` owns the `with` and calls the cell's `build()` closure inside it.
- **The trace-body closure must be a named function, not a `lambda`.** Under pipeline/tensor parallelism nnsight *source-serializes* the closure to the vLLM worker, and a `lambda` has no recoverable source on Py3.10 — it would reach the worker empty. So every cell passes a `def build()` (or `def capture()`/`def step()`/…) to `be.run`/`be.patch`/`be.attribute`/`be.generate`, never a `lambda`. Lambdas happen to work on hf/vllm_async (no serialization there), which is exactly why the convention is enforced uniformly rather than per-backend.
- `docs/design.md` is the living design doc; code comments cite its sections (e.g. `§12.2`) — keep those citations accurate when changing design-relevant code. Measured results go in `docs/findings.md`; the per-context primitive status inventory lives only in `docs/interp-methods-catalog.md` (single copy, so lists can't diverge).
- The corpus deliberately includes non-portable workloads as frontier markers — an `ERROR` cell on vLLM is a *result*, not a gap to "fix" by deleting the cell. Methodologies often come as matched pairs: the naive port (frontier marker) and the documented-correct form for the backend.
- **A TP/PP divergence is a finding to classify and document, NOT a cell bug to patch.** By default tensor/pipeline parallelism is assumed to share the *same behavior and same interface* as single-GPU — identical intervention code runs identically on 1 and N GPUs. When `bench.py --pp/--tp` finds a cell that is correct on single-GPU but `DIVERGENT` (the GT2 oracle scores parallel-vs-single vLLM, not vs HF — see design §12.2), that is the benchmark *working*. Classify the divergence as either an interface change users must be made aware of, or — the **default for parallelism** — a translation gap **nnsight** should fix, because an interp user must not have to handle the underlying sharding/distribution. Do NOT rewrite the cell to work around it (that hides the finding and pushes distribution onto users). E.g. `head.weight[token_id]` is correct single-GPU code; under TP vLLM vocab-shards `lm_head`, so the same index returns a shard-local row → an nnsight TP-transparency finding, not a steering-cell fix. (Distinct from a cell using a *known-wrong* pattern like reading the plain residual on a fused-residual family, which IS a suite bug to fix in the cell.)
- Tests are no-GPU by design: cell logic, oracle, execute/score invariants run against fake backends. GPU behavior is exercised by `scripts/bench.py`, not the test suite.
- `serve.cli --host 0.0.0.0` executes pickled Python from the network — only safe on the trusted docker compose bridge; never expose port 6677.

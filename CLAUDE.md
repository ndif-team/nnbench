# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

nnbench: a **systems performance + coverage benchmark** for interpretability workloads run through nnsight across serving backends (HuggingFace vs vLLM). It measures whether a workload **runs**, **runs correctly** (numerical equivalence vs an HF reference), and **runs fast** — it is *not* a faithfulness benchmark. The headline deliverable is the applicability map, and the dangerous state it exists to catch is `SILENTLY_WRONG`: runs with no error but produces wrong numbers (e.g. the portable logit-lens on vLLM-Llama drops half the dual residual stream).

## Environment & commands

Everything runs in the `nnsight-serve-test` conda env: **vLLM 0.15.1** + nnsight editable from `/disk/u/zikai/nnsight/src` (the `dev` branch). Two caveats:

- **Version skew**: the nnsight `dev` branch targets vLLM **0.19.1**, while this env pins **0.15.1**. Running the benchmark here is correct, but never characterize *nnsight-side* vLLM behavior from this env alone — verify under the `ndif-dev` env (vLLM 0.19.1) with `PYTHONPATH=/disk/u/zikai/nnsight/src` prepended (its editable nnsight points at a different worktree; PYTHONPATH wins). Confirm with `python -c "import nnsight,os; print(os.path.dirname(nnsight.__file__))"`.
- **`conda run` mishandles signals** on long vLLM runs (false timeouts, swallowed SIGABRT). For anything GPU/vLLM, prefer the env's python directly: `/disk/u/zikai/anaconda3/envs/nnsight-serve-test/bin/python`. Always wrap nnsight/vLLM runs in `timeout`, and afterwards verify no orphan processes remain and GPU memory is freed (an unguarded HF run once hung 80 min holding the GPU).

```bash
# Unit tests — all no-GPU (fake backends, torch-only logic)
conda run -n nnsight-serve-test python -m pytest tests/ -q
conda run -n nnsight-serve-test python -m pytest tests/test_sweep.py -q     # one file
conda run -n nnsight-serve-test python tests/test_sweep.py                  # files also self-run via _run_all()

# Benchmark (needs GPU) — env python directly + timeout, not conda run (see caveats above)
PY=/disk/u/zikai/anaconda3/envs/nnsight-serve-test/bin/python
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec steering_gpt2
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec all --backends hf
# Specs live in isb/specs/ (logit_lens_gpt2, logit_lens_llama, steering_gpt2, activation_patching_gpt2,
# ablation_gpt2, ...). Llama specs need HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.

# Micro tier (Level 0/1 primitive map) — ONE backend per process, always under `timeout`
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/micro.py --backend hf

# Client/server (vllm_serve) split — see docker/README.md
GPU=5 docker/run_vm.sh                      # after dumping refs with bench.py --dump-refs / --dump-ctl-refs
```

vLLM EngineCore uses spawn: every entrypoint must run under an `if __name__ == "__main__"` guard.

## Architecture

The core unit is a **cell**: one explicit function per `(methodology, family, backend)`, registered with `@cell(...)` in `isb/methodologies/` (registry in `isb/methodologies/registry.py`). Variances (prompts, layers, idiomatic-vs-portable formulation) are runtime **params** to the cell, not separate registrations. This flatness is deliberate — an earlier "Resolver" abstraction that *generated* intervention code from declarations was killed (design.md §11–12); the leveled primitive model in design.md §3 only *indexes* cells, never constructs them. Do not reintroduce a construction layer. Every `vllm_*` variant cell (`vllm_serve`, `vllm_sync`, `vllm_pp`, …) falls back to the `vllm_async` cell automatically (same intervention code; the variant difference — over-HTTP, in-process-sync, pipeline/tensor-parallel — lives entirely in the backend object).

Data flow for one spec (`scripts/bench.py` → `isb/sweep/driver.py:run_sweep`):

1. **Spec** (`isb/specs/`): a `CellConfig` + `Workload`s (interactive / batched) per methodology. Batching is a *coverage axis* — each workload is oracle-checked in its own regime, not just timed.
2. **Backend** (`isb/backends/`): `be` objects — `hf` (the per-family control), `vllm_async` (in-process system under test), `vllm_serve` (over-HTTP). One model load per backend, amortized across all tasks; an intervention error is isolated (engine survives, later tasks still run).
3. **Oracle** (`isb/oracle/equivalence.py`): each non-HF cell is scored against the HF cell of the same family via top-1 token agreement + softmax total-variation distance. Precision divergence is disambiguated by an fp32 rerun (in-process) or cached fp32-vLLM refs (`--ctl-refs`, for the GPU-less serve client) → `SUPPORTED_DEGRADED` vs `SILENTLY_WRONG`.
4. **Perf** (`isb/perf/`): warm timing (warmup + N trials, CUDA-synced, median±std, peak mem) — correctness is verified in the same warm/batched regime perf is measured in.
5. **Report** (`isb/report/`): applicability map (`AppState` in `isb/states.py`) + performance table.

The **micro tier** (`isb/micro/`) probes individual nnsight primitives per backend. A probe that times out is recorded `HANG` and **aborts that backend's remaining probes** — the stuck thread still owns the engine loop, so anything after it would measure a poisoned engine. Probe registration order is safest-first for this reason.

## Conventions & constraints

- **The trace body must live in the same frame as `with model.trace(...)`** — this nnsight dev branch compiles the captured body, so splitting it across a generator/`@contextmanager` yields an empty body and deadlocks. `be.run` owns the `with` and calls the cell's `build()` closure inside it.
- **The trace-body closure must be a named function, not a `lambda`.** Under pipeline/tensor parallelism nnsight *source-serializes* the closure to the vLLM worker, and a `lambda` has no recoverable source on Py3.10 — it would reach the worker empty. So every cell passes a `def build()` (or `def capture()`/`def step()`/…) to `be.run`/`be.patch`/`be.attribute`/`be.generate`, never a `lambda`. Lambdas happen to work on hf/vllm_async (no serialization there), which is exactly why the convention is enforced uniformly rather than per-backend.
- `docs/design.md` is the living design doc; code comments cite its sections (e.g. `§12.2`) — keep those citations accurate when changing design-relevant code. Measured results go in `docs/findings.md`; the per-context primitive status inventory lives only in `docs/interp-methods-catalog.md` (single copy, so lists can't diverge).
- The corpus deliberately includes non-portable workloads as frontier markers — an `ERROR` cell on vLLM is a *result*, not a gap to "fix" by deleting the cell. Methodologies often come as matched pairs: the naive port (frontier marker) and the documented-correct form for the backend.
- Tests are no-GPU by design: cell logic, oracle, driver invariants run against fake backends. GPU behavior is exercised by `scripts/bench.py`, not the test suite.
- `serve.cli --host 0.0.0.0` executes pickled Python from the network — only safe on the trusted docker compose bridge; never expose port 6677.

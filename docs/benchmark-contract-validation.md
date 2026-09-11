# Benchmark contract validation

Date: 2026-09-10

The implementation follows [design §12.13](design.md#1213-current-execution-and-artifact-contract).
The design was updated before implementation on
`codex/benchmark-contract-consistency`. The nnsight 0.8 dependency pin is retained.

## Acceptance evidence

| Planned contract | Implementation and regression evidence |
|---|---|
| One effective call | `test_call_binding.py` covers real signature binding, default application, required/unknown arguments, partial functions, profile-bound generic cells, positional-only defaults, and dataset/task/generation precedence. All 24 existing specs bind on HF and vLLM with matching semantic settings. |
| Concrete case requirements | `test_case_requirements.py` observes actual cell branches for writes, gradients, patching, ablation targets, and zero-strength steering. Custom hooks and explicit template-only fallback are covered. |
| Ownership and timing | `test_sweep.py`, `test_timing.py`, and `test_worker_execution.py` verify fresh nested values for warmups, trials, aggregation, baselines, references, and effects. A controlled clock asserts that preparation occurs outside the timed interval. |
| Complete comparison identity | `test_coordinates.py` and `test_score_identity.py` reject changed case settings, input content/order, regime controls, baseline/effect configuration, and a mismatched current scoring spec. Timing-only changes compare while remaining recorded. Incomplete current identity cannot establish equivalence. |
| Current file boundaries | `test_job_reader.py` and `test_jobs.py` cover saved custom descriptors, explicit opt-outs, deferred runtime classification, malformed jobs, unsupported versions, and atomic failure cleanup. |
| Resolved choices | `test_resolved_choices.py` exercises the real logit-lens, steering, and Jacobian-lens resolution sites across GPT-2 and Llama profiles. It covers explicit overrides, owned observations, nested context cleanup, and inconsistent reports. Worker integration checks task and auxiliary observations through tensor, JSON, and manager paths. |
| Structured failure reporting | Malformed jobs and restoration failures produce atomic failure JSON. Invalid calls retain attempted/effective settings and their failure stage. Empty, missing, nonfinite, and incompatible effect outputs remain isolated failures. |
| Consistent saved artifacts | `test_worker_execution.py` runs the production worker and executor with a CPU fixture backend. Task and auxiliary metadata survive tensor artifact, JSON result, and JSON-only manager loading. |
| Independent workers | `test_independent_worker.py` runs a worker that imports no nnbench helpers. A v1-only reader clearly rejects v2. The three CPU Docker tests cover successful execution, crash, timeout, and project cleanup. |
| Usable result pages | Manager tests cover pending/failing jobs, case records, incomplete-current-record warnings, and complete static export. Failed/incomplete effect checks render safely with escaped diagnostic text. |
| Lightweight host | Host spec discovery, launcher, and manager import without torch, nnsight, or vLLM. Both configured backend directories and all 24 specs remain discoverable. |
| CausaLab alignment | The source checker verifies the existing vocabulary snapshot against upstream checkout `8696e04bfb06a169defe1bf563d8aeef992f85cd`. |

## Validation commands

Full CPU suites: **458 passed, 3 opt-in Docker tests skipped** on each of Python **3.11.15** and
**3.12.12**. The separate CPU-only Docker lifecycle run passed **all 3 tests**, using the existing
`isb-nnsight-hf:local` image and Python 3.11.15 as the host interpreter. The Python 3.12 host
reported unavailable NVML during CUDA capability probes; no tests required a GPU.

Changed/new Python files pass Ruff. `git diff --check`, CLI discovery, host import isolation, and
pinned vocabulary validation pass. Manager static export is exercised by worker and
failure-reporting integration tests. These counts replace the earlier Python-3.11-only report.

```bash
python -m pytest -q -p no:cacheprovider
ISB_DOCKER_TESTS=1 python -m pytest tests/test_jobs_docker.py -q -p no:cacheprovider
python scripts/check_causalab_alignment.py /path/to/pinned/causalab --check
python scripts/bench.py list backends
python scripts/bench.py list specs
git diff --check
```

Ruff is run over all changed and newly added Python files. Existing unrelated repository lint
findings are outside this refactor.

## Intentional compatibility changes

- Python authoring uses `CellConfig(regimes=...)` and `TaskSpec(label, params)`; `(params, label)`
  task tuples remain accepted. Historical aliases and the task tuple iterator are retired.
- Jobs use wire version 2 and run coordinates use schema 3. Other versions are rejected with a
  rerun instruction. The follow-up removes historical readers and the obsolete v1 fixture;
  existing run files are left untouched.
- Methodology templates remain torch-free. Optional case-description hooks are registered beside
  the cell implementations and receive already-bound parameters.
- Timing keeps trial preparation outside the measured interval. Effect checks use the same full
  aggregation regime as the measured case.

## Boundary of this validation

These checks establish framework behavior, current file validation, and preservation of benchmark
findings. The CPU fixtures do not measure GPT-2 model performance or prove GPU backend support.
The full GPT-2/nnsight 0.8 sweep and its benchmark artifact page remain a separate run requiring
an uncontended GPU allocation. Unsupported cells must remain visible in that page.

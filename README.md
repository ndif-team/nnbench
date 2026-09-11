# nnbench

A systems performance and coverage benchmark for interpretability workloads run through
[nnsight](https://nnsight.net) across serving backends. nnbench measures whether an intervention
runs, whether its outputs agree across backends, and its latency, throughput, memory use and
overhead. Numerical comparison exposes silent errors that successful execution alone can miss.

The core unit is an explicit cell for a methodology, model family and backend. Cases vary the
intervention parameters, implementation choices and input regime. Scientific faithfulness is a
separate question studied by frameworks and benchmarks such as CausaLab, CausalGym and InterpBench.

## Case descriptions and CausaLab alignment

Each benchmark case separates three concerns:

- `InterventionSpec`: intervention semantics and component-level requirements.
- `TaskSpec`: a case label and the params its cell receives; the methodology's `InterventionSpec`
  names which of them are semantic values and which are realization selectors, such as residual
  spelling or bounded iteration.
- `ExecutionRegime`: input units, interactive/batched/generation shape, decode length and aggregation.

These descriptions index explicit cell implementations and their results. CausaLab's current
protocol/engine stack supplies the shared vocabulary. The checked-in
[vocabulary snapshot](isb/causalab_vocabulary.json) pins upstream commit
[`8696e04`](https://github.com/goodfire-ai/causalab/tree/8696e04bfb06a169defe1bf563d8aeef992f85cd),
including canonical component names and deprecated aliases. Pyvene and nnterp are historical
context; neither is a runtime dependency of this alignment.

The worker binds the final settings to each cell's Python signature, including its defaults.
Case-description hooks beside the method code record concrete read/write/gradient requirements;
custom saved descriptions without a matching hook retain explicit template-level scope.
Cases record these descriptions, bound semantic and realization values, and the vocabulary
reference with their results. Vocabulary membership establishes a
valid name; measured backend coverage comes from benchmark runs. nnbench extensions identify
behavior beyond CausaLab's document contract, such as writes at every decode step.

Custom methods supply an `InterventionSpec` or an explicit `protocol_absence_reason`. The worker
isolates nested execution parameters and provenance snapshots, preparing trial configuration
outside the timed interval. Cells report automatically resolved choices separately in
`resolved_params`, preserving the original bound arguments.

Readers accept job wire version 2 and coordinate schema 3; older formats must be rerun.
Correctness comparisons require matching input content, case/regime settings, baseline/effect
configuration, and model-load options. Timing-only differences are allowed and stay recorded.
The [current design and acceptance plan](docs/design.md#1213-current-execution-and-artifact-contract)
defines these boundaries and the extension points for methods and independent backend workers.

See [the alignment audit](docs/causalab-portability-audit.md) for the current engine architecture,
scope and update procedure. Verify the pinned source against a checkout at that revision:

```bash
python scripts/check_causalab_alignment.py /path/to/causalab --check
```

## Results and scoring

The launcher freezes one experiment and its inputs, then runs each selected backend in an
independent container. Workers save outputs, timing and provenance; a separate scorer compares
the saved artifacts. Select the reference explicitly with `--reference`, normally `nnsight-hf`.

| State | Meaning |
|---|---|
| `RAN` | Produced an output; no comparison verdict was assigned, including reference rows |
| `SUPPORTED` / `SILENTLY_WRONG` | Matches / fails the selected correctness reference |
| `EQUIVALENT` / `DIVERGENT` | Matches / differs under `--comparison equivalence` |
| `ERROR` | The cell failed |
| `NO_REFERENCE` / `INVALID_REFERENCE` | Reference output is unavailable / unsuitable for judgment |
| `INCOMPATIBLE` | Source, model or tokenizer identity prevents a valid comparison |
| `JOB_FAILED` | Container execution or artifact validation failed |

The numerical oracle uses top-1 agreement and softmax total-variation distance. Batched
correctness is checked against per-prompt reference outputs. The main runner uses the requested
comparison policy; legacy precision-control scoring can additionally classify
`SUPPORTED_DEGRADED`.

Warm timing uses warmup plus repeated trials, CUDA synchronization, median and standard deviation.
Reports include peak GPU memory, overhead against a no-intervention baseline and regime-specific
throughput. Timing and numerical verdicts are recorded separately.

## Implemented scope

Specs cover logit lens, steering, ablation, activation patching, attention-pattern reads,
attribution patching, generation steering and patching, DAS, Jacobian collection and Jacobian
lens. Model configurations include GPT-2, Llama-family models, Qwen2.5, Qwen3.5 and Nemotron hybrids.

The [spec registry](isb/specs/__init__.py) is the authoritative inventory. Each spec defines the
available method/model combinations; the inventory is not a claim that every combination passes.
`--spec all` selects the small default corpus. Larger model configurations are selected by name.
See [measured findings](docs/findings.md) and the
[primitive status inventory](docs/interp-methods-catalog.md) for results tied to tested stacks.

## Running it

Requirements: Docker Engine, Compose v2, NVIDIA Container Toolkit for the bundled GPU backends,
and host Python with CPU PyTorch for scoring. Each backend directory owns its image, dependencies
and entrypoint.

```bash
python scripts/bench.py list backends
python scripts/bench.py list specs
python scripts/bench.py build nnsight-hf nnsight-vllm

# Compare vLLM with an explicit HF reference.
python scripts/bench.py run --spec logit_lens_gpt2 --data factual:2 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0

# Collect the small default corpus on HF.
python scripts/bench.py run --spec all --backends nnsight-hf --gpu 0

# Re-score an existing run.
python scripts/bench.py score runs/REPLACE_WITH_RUN_ID
```

Each invocation creates a unique directory under `--out` (default `runs/`). `--timeout` limits
each backend job; `--strict` also returns a failure exit code for unsuccessful cell verdicts.
Model downloads populate a persistent cache. Offline flags are useful once that cache is stocked.
See [the backend guide](backends/README.md) for configuration, model access and custom backends.

Standalone micro/performance and execute/score tools retain their own entrypoints and
environment requirements.

## Browsing results

```bash
python scripts/manager.py --dir runs --port 6688
python scripts/manager.py --dir runs --export results.html
```

Open `http://127.0.0.1:6688` or share the self-contained HTML export. The site reads saved JSON
reports; it leaves tensor artifacts and scoring untouched. See
[the result-site guide](docs/result-site.md) for legacy imports and collection management.

## Validation

No-GPU tests exercise cell logic, numerical scoring, configuration ownership, protocol metadata,
serialization and runner contracts:

```bash
python -m pytest tests/ -q

# Optional CPU-only container lifecycle tests; requires the built HF image.
ISB_DOCKER_TESTS=1 python -m pytest tests/test_jobs_docker.py -q
```

Harness tests and vocabulary checks establish implementation contracts. Model execution and
cross-project numerical parity require separate runs.

## Layout

```text
isb/
  protocol.py              torch-free methodology templates and parameter classifications
  causalab_vocabulary.json pinned upstream vocabulary and source checksums
  methodologies/           explicit cells and case-requirement description hooks
  backends/                model-access and execution infrastructure
  specs/                   benchmark case definitions
  sweep/                   spec types and shared cell execution; legacy scoring
  jobs/                    frozen experiments, container lifecycle and artifact scoring
  oracle/                  numerical comparison
  perf/                    timing and microbenchmarks
  manager/                 saved-report browser and HTML export
backends/                  independent NAME/{compose.yml,Dockerfile,run.py} packages
scripts/                   benchmark CLI, alignment checker, manager and standalone tools
docs/                      design, alignment audit, guides and measured findings
```

See [the living design](docs/design.md) for architectural decisions.

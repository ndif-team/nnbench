# nnbench

A systems performance and coverage benchmark for interpretability workloads on LLM serving
engines. nnbench runs the same workloads on [vLLM](https://github.com/vllm-project/vllm) through
five interpretability systems ([nnsight](https://nnsight.net), vLLM-Lens, interp-engine, the
TransformerLens vLLM bridge and vLLM-Hook) and on HF transformers through nnsight. Plain vLLM of
the same version, serving the same prompts with no intervention, is the floor a system's overhead
is measured against. For every workload nnbench records whether the system can express it, whether
its output agrees with a reference, and its latency, throughput and overhead under single requests,
offline batches and online load.

Measured numbers for the pinned releases are in [docs/results.md](docs/results.md).

The core unit is an explicit cell for a methodology, model family and backend. Cases vary the
intervention parameters, implementation choices and input regime. Scientific faithfulness is a
separate question studied by frameworks and benchmarks such as CausaLab, CausalGym and InterpBench.

## Implemented scope

The workloads (`bench.py list workloads`) cover the logit lens and the Jacobian lens, residual
readout, steering, ablation, activation patching, attention-pattern reads, attribution patching,
DAS, Jacobian collection, activation harvesting, and, during generation, steering, patching and
capture. Two more sweep the number of steered or exported layers. The models
(`bench.py list models`) are GPT-2, SmolLM2-135M, Qwen2.5 (7B, 14B, 72B and the 72B AWQ
quantization), Qwen3 (8B, 32B), Qwen3.5-4B, Llama-3.3-70B and Nemotron 3 Nano (4B, 30B).

Definitions come in four independent layers ([design §12.16](docs/design.md)):

- a **workload** ([`isb/workloads/`](isb/workloads/), one per methodology plus the two layer-count
  sweeps) holds the intervention semantics, data, regimes and effect check, and names no model and
  no system;
- a **model** ([`isb/models.py`](isb/models.py)) is a checkpoint with its architecture, layer
  count and load options; workload layers are depth fractions resolved against it;
- a **realization** is how one system spells a row (nnsight's in-place vs replacement write, the
  naive plain-residual port), declared on that system's cell registration;
- a **suite** ([`isb/suites.py`](isb/suites.py)) is a run recipe: workloads × models, backends,
  reference, data size and realizations.

A case is named `<workload>/<model>`. A row a model cannot express (attention ablation on a
single-mixer block) is reported as not applicable. The inventory is not a claim that every
combination passes.

The suites (`bench.py list suites`) are `smoke` (every workload on GPT-2 and the logit lens on
SmolLM2-135M, nnsight on HF and vLLM), `cross-system`, `parallel` (the cases for comparing a
multi-GPU config with its single-GPU config), `nemotron` (Nemotron 3 Nano 4B), `fitted-jlens`
(Qwen3.5-4B with its published Jacobian lens), the request-pattern suites (`nnsight-load-offline`,
`nnsight-load-concurrent`, `nnsight-load-open`), the nnsight release comparisons
(`nnsight-versions`, `nnsight-large-*`), the Llama-3.3-70B suites (`llama70b-*`), and the
layer-count and transfer-volume sweeps. Each entry of [`isb/suites.py`](isb/suites.py) says what
it measures.

See [measured results](docs/results.md) for the numbers on the pinned releases,
[measured findings](docs/findings.md) for the dated records of earlier runs, and the
[primitive status inventory](docs/interp-methods-catalog.md) for per-primitive status on nnsight.

## Systems

Each system is a directory under `backends/` with one config per variant. The table gives the
releases each Dockerfile and `configs.yml` pin and the main configs; `bench.py list backends`
prints every config.

| System | Pinned release | vLLM | Main configs |
|---|---|---|---|
| nnsight on vLLM | 0.8.1 in the `0-8-1` configs; 0.7.0 in `0-7`, 0.5.15 in `0-5-15` | 0.27.1; 0.15.1; 0.13.0 | `nnsight-vllm/0-8-1` (the sync engine, nnsight's documented `LLM` path), `0-8-1/async` (`AsyncLLM`), `0-8-1/opt` (installed edits), `0-8-1/opt/taps` (CUDA-graph taps) |
| nnsight on HF transformers | 0.8.1, 0.7.0, 0.5.15 and 0.4.6 (`0-4-6`), on transformers 5.12.1 (5.19.0 in `tp2`, `tp4`; 4.51.3 in `0-4-6/era`) | | `nnsight-hf/0-8-1` (bf16, SDPA attention); `0-8-1/gpu2`, `0-8-1/gpu4` (layers placed across GPUs, `device_map=auto`); `0-8-1/tp2`, `0-8-1/tp4` (transformers tensor parallelism) |
| vLLM-Lens | 1.2.1 | 0.19.1 | `vllm-lens/default`, `vllm-lens/opt`, `vllm-lens/async` |
| interp-engine | 1.12.0 | 0.28.0 | `interp-engine-vllm/default` (hooked), `interp-engine-vllm/static` (`vllm-static`) |
| TransformerLens 4, vLLM bridge | 4.0.0 | 0.20.2 | `transformer-lens-vllm/default`, `transformer-lens-vllm/batched` (`enable_batching`), `transformer-lens-vllm/opt` (headroom, see below) |
| vLLM-Hook (IBM) | commit `0e34fdd` (`vllm_hook_plugins` 0.2.0) | 0.19.1 | `vllm-hook/default` (rpc storage), `vllm-hook/disk` (`disk-st-async`), `vllm-hook/qk` (the Q/K probe worker) |
| Plain vLLM | | 0.15.1, 0.19.1, 0.20.2, 0.27.1, 0.28.0 | `vllm-plain/<version>` (vLLM's default mode: torch.compile and CUDA graphs) and `vllm-plain/<version>/eager` |

The bare `nnsight-vllm/default` and `nnsight-hf/default` configs build nnsight 0.8.0rc1 (commit
`260c555`, on vLLM 0.19.1); the `smoke` and `cross-system` suites use them. `nnsight-hf/fp32`,
`nnsight-hf/sdpa` and `nnsight-hf/eager` (eager attention, the attention-pattern reference) run
that image with other options. A released version is selected by its config prefix. nnsight,
vLLM-Lens and vLLM-Hook also have `async` configs (vLLM's `AsyncLLM`, for concurrent and open-loop
clients). Every system on vLLM has configs with tensor parallelism over two or four GPUs. Their
names end in `tp2` or `tp4`: `nnsight-vllm/0-8-1/tp2` beside `nnsight-vllm/0-8-1`, and
`vllm-lens/tp2` beside `vllm-lens/default`.

Plain vLLM at a system's vLLM version is its overhead floor, in vLLM's default mode and, for a
system that runs the engine eager, in the eager mode as well ([Running it](#running-it)). vLLM
0.13.0 (nnsight 0.5.15) has no plain config. Prefix caching is off on every vLLM engine: the
config catalogs disable it, and the TransformerLens bridge turns it off itself. A workload a system
cannot express is `UNSUPPORTED` with the reason from its source; that is a result.

The `cross-system` suite runs logit lens, steering, ablation, generation steering, activation
patching, activation harvesting and generation capture on Qwen2.5-7B-Instruct through the default
configs of nnsight, vLLM-Lens, interp-engine and TransformerLens, against `nnsight-hf/default`,
with the default-mode plain-vLLM floors of their three vLLM versions (`vllm-plain/0-19-1`,
`0-20-2`, `0-28-0`).

A score measures the system, not the cell author. Each system is scored by its general cell and by
its fastest realization inside its public interface: documented engine modes, documented storage,
and compositions of its documented API. A faster form that needs the system's internals or redoes
its work is reported as headroom, never as its score. The rule is in
[writing workloads §4a](docs/writing-workloads.md); measured results, including the fairness
controls, are in [the findings](docs/findings.md).

## Running it

Requirements: Docker Engine, Compose v2 and the NVIDIA Container Toolkit for the bundled GPU
backends (or Apptainer, below), and a host Python with PyYAML and CPU PyTorch for the launcher and
the scorer. Each system directory under `backends/` owns its image and implementation; each config
inside it (`<system>/<config>`) owns its options.

```bash
python scripts/bench.py list backends
python scripts/bench.py list workloads   # also: models, suites
python scripts/bench.py build nnsight-hf/0-8-1 nnsight-vllm/0-8-1

# Compare vLLM with an explicit HF reference.
python scripts/bench.py run --workload logit_lens --model gpt2 --data factual:2 \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 --reference nnsight-hf/0-8-1 --gpu 0

# The small default corpus: every workload on gpt2 and the logit lens on SmolLM2-135M, nnsight on
# HF and vLLM (the `default` images).
python scripts/bench.py build nnsight-hf/default nnsight-vllm/default
python scripts/bench.py run --suite smoke --gpu 0

# Another system beside plain vLLM of its release, which adds the overhead columns. vLLM-Lens
# runs the engine eager, so the run takes the eager floor beside the default one (same image).
python scripts/bench.py build vllm-lens/default vllm-plain/0-19-1
python scripts/bench.py run --workload steering --model qwen2.5-7b \
  --backends nnsight-hf/0-8-1 vllm-lens/default vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0

# Re-score an existing run.
python scripts/bench.py score runs/REPLACE_WITH_RUN_ID
```

Each invocation creates a unique directory under `--out` (default `runs/`) holding the frozen
inputs, each job's artifacts and `report.json`. The backends of a run execute one after another,
one container per backend and case. `--gpu` takes one GPU index: a config that reserves several
GPUs (`tp2`, `gpu4`) uses that index and the ones after it, so `--gpu 4` with a `tp4` config runs
on GPUs 4 to 7. `--timeout` limits each backend job. The command exits 1 when a job fails; with
`--strict` it also exits 1 when a row is `UNSUPPORTED`, `ERROR` or `INVALID_OUTPUT`. `--regimes`
keeps a subset of a workload's regimes and `--labels` a subset of its rows. A `--workload` run
without `--suite` runs every realization the cells declare, and `--realizations default` keeps
each system's documented form only; a suite sets its own realizations. Model downloads populate a
persistent cache; with the checkpoints in it, `HF_HUB_OFFLINE=1` keeps a run off the network. See
[the backend guide](backends/README.md) for configuration, model access and custom backends.

A report carries two overhead ratios against plain vLLM, each the row's latency divided by the
latency of the no-intervention request of a plain-vLLM job in the same run.
`overhead_vs_plain_vllm` uses the plain job whose engine ran the same vLLM version, the same mode
(torch.compile and CUDA graphs on or off) and the same tensor-parallel size.
`overhead_vs_plain_vllm_default` uses the plain job in vLLM's default mode at that version and
size, and is set for rows whose engine ran a different mode. `vllm-plain/<version>` in `--backends`
is the default-mode floor; a system that runs the engine eager needs
`vllm-plain/<version>/eager` as well and then carries both ratios. The `opt/taps` configs of
nnsight and `interp-engine-vllm/static` run CUDA graphs without torch.compile, a mode no
`vllm-plain` config runs, so their rows carry the second ratio only.

### Request patterns

Beside the single-call regimes a workload declares (interactive, batched, generation), three load
patterns measure throughput and latency under load ([request patterns](docs/request-patterns.md)):

```bash
# Offline: 256 units per pass, in calls of 1, 16, 64 and all units.
python scripts/bench.py build vllm-plain/0-27-1
python scripts/bench.py run --workload steering --model qwen2.5-7b --regimes offline \
  --offline 256:1,16,64,all \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0

# Closed loop: 128 single-unit requests from 1, 4, 16 and 64 clients.
# Open loop: 128 single-unit requests arriving at 4, 16 and 64 per second.
python scripts/bench.py run --workload steering --model qwen2.5-7b --regimes concurrent open \
  --concurrent 128:1,4,16,64 --open 128:4,16,64 \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1/async --reference nnsight-hf/0-8-1 --gpu 0
```

The `nnsight-load-*` suites are these patterns as recipes. The `async` configs run vLLM's
`AsyncLLM`, which takes requests together; HF and the sync engine serve one request at a time.

### Hosts without Docker (Apptainer)

On HPC hosts such as NCSA Delta, set `ISB_LAUNCHER=apptainer` (or pass `--launcher apptainer`).
The runner reads the same `compose.yml` and Dockerfile: `build` translates the Dockerfile into an
Apptainer definition and a `.sif` image, and `run` maps the `runner` service onto
`apptainer exec --nv`. Images, definitions and named volumes live under `ISB_APPTAINER_DIR`
(default `.apptainer/`); the model cache is `$ISB_APPTAINER_DIR/volumes/isb-model-cache`.

```bash
export ISB_LAUNCHER=apptainer ISB_APPTAINER_DIR=$PWD/.apptainer
python scripts/bench.py build nnsight-hf/fp32 nnsight-vllm/default vllm-lens/default vllm-plain/0-19-1
HF_HUB_OFFLINE=1 python scripts/bench.py run --suite cross-system --workload steering \
  --backends nnsight-hf/fp32 nnsight-vllm/default vllm-lens/default \
             vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/fp32 --gpu 0
```

`slurm/delta-build.slurm` builds images and `slurm/delta-compare.slurm STAGE` runs the comparison
stages (`fair`, `optimal`, `vllm-hook`, `micro`, ...) as batch jobs. Their `#SBATCH` headers
(account, partition, node exclusions) and the storage paths in their bodies are those of one Delta
allocation; set them for yours. [`docs/delta-handoff.md`](docs/delta-handoff.md) records a first
run on Delta step by step (2026-09-30).

### Other tools

- `scripts/profile_phases.py` splits each row's latency into phases (request transfer, engine
  steps, interpretability-system hooks, readout) on a vLLM backend config, one Docker container per
  case and config. `--combine` then splits each profile against the default-mode plain-vLLM floor
  profiled into the same directory.
- `scripts/audit_runs.py` checks a set of run directories before their numbers are used: every job
  completed, one source commit, under 100 MiB in use on the job's first GPU before it started, no
  row in a hard failure state, and, in a run that includes a `vllm-plain` config, a same-mode floor
  (`overhead_vs_plain_vllm`) for every timed vLLM row. Rows of the `opt/taps` and `static` configs
  have no same-mode floor, so the audit lists them as errors in such a run.
- The perf microbenchmark (`scripts/perf.py`) times single operations (read, steer, Q/K) per
  system against plain vLLM. Each cell runs in its own process; `ISB_PY_<system>` selects the
  interpreter for a system's cells, for example an `apptainer exec` wrapper around its backend
  image.
- The construct probes (`scripts/micro.py`) and the standalone executor and scorer
  (`scripts/execute.py`, `scripts/score.py`) keep their own entrypoints and environment
  requirements.

## Results and scoring

The launcher freezes one experiment and its inputs, then runs each selected backend in an
independent container. Workers save outputs, timing and provenance; a separate scorer compares
the saved artifacts. Select the reference explicitly with `--reference`, normally nnsight on HF
(`nnsight-hf/0-8-1`).

| State | Meaning |
|---|---|
| `RAN` | Produced an output; no comparison verdict was assigned, including reference rows |
| `SUPPORTED` / `NUMERICAL_MISMATCH` | Meets / exceeds the selected numerical thresholds; neither establishes general correctness |
| `UNSUPPORTED` | The system's interface cannot express the row (the cell records the reason from the system's source), or no cell is registered for the combination |
| `INVALID_OUTPUT` | Output structure or finiteness check failed; timing is diagnostic only |
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
Reports include regime-specific throughput, overhead against a no-intervention call on the same
system, overhead against plain vLLM (the `vllm-plain` configs in the same run, see
[Running it](#running-it)), and the benchmark process's peak GPU allocation (`peak_mem_mb`). On
vLLM the model runs in the engine's processes, whose peak is recorded only when the run sets
`ISB_WORKER_MEMORY=1`. Timing and numerical verdicts are recorded separately: a
`NUMERICAL_MISMATCH` row keeps its timing, and only an execution failure or a demonstrated contract
violation removes a row from performance comparison.

## Browsing results

```bash
python scripts/manager.py --dir runs --port 6688
python scripts/manager.py --dir runs --export results.html
```

Open `http://127.0.0.1:6688` or share the self-contained HTML export. The site reads saved JSON
reports; it leaves tensor artifacts and scoring untouched. See
[the result-site guide](docs/result-site.md) for legacy imports and collection management.

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

## Validation

No-GPU tests exercise cell logic, numerical scoring, configuration ownership, protocol metadata,
serialization and runner contracts. They import the cells, so the environment needs nnsight
(which brings PyTorch and transformers), PyYAML and pytest:

```bash
python -m pytest tests/ -q
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
  workloads/               a methodology's rows, data, regimes and effect check; two layer-count sweeps
  models.py                registered checkpoints: architecture, layer count, load options
  suites.py                run recipes: workloads x models, backends, reference, load patterns
  sweep/                   spec types and shared cell execution; legacy scoring
  jobs/                    frozen experiments, container lifecycle and artifact scoring
  oracle/                  numerical comparison
  perf/                    timing, load patterns, the phase profiler and microbenchmarks
  micro/                   construct probes for individual nnsight primitives
  report/                  applicability map and performance table
  manager/                 saved-report browser and HTML export
backends/                  <system>/{Dockerfile,compose.yml,configs.yml,backend.py,cells.py}; configs.yml lists the images and
                           configs; the nnsight systems have no cells.py (their cells are in isb/methodologies/)
slurm/                     batch scripts for the Apptainer path on Slurm clusters
scripts/                   benchmark CLI, alignment checker, manager and standalone tools
docs/                      results, design, guides and dated records
docker/                    the preceding shared-Compose runner; `bench.py` does not use it
configs/                   workload files of the June 2026 prototype; no code reads them
```

## Documentation

| Document | Content |
|---|---|
| [docs/results.md](docs/results.md) | Measured support, latency, throughput and overhead for the pinned releases |
| [backends/README.md](backends/README.md) | The backend file contract, configs, model access, adding a system |
| [docs/writing-workloads.md](docs/writing-workloads.md) | How to write a workload and a cell, and what to run before pushing one |
| [docs/request-patterns.md](docs/request-patterns.md) | The single-call regimes and the offline, closed-loop and open-loop patterns |
| [docs/design.md](docs/design.md) | The living design: contracts, architectural decisions and their reasons |
| [docs/findings.md](docs/findings.md) | Dated measurement records, each tied to the stack it ran on |
| [docs/interp-methods-catalog.md](docs/interp-methods-catalog.md) | Interpretability methods and per-primitive status on nnsight |
| [docs/result-site.md](docs/result-site.md) | The result browser and its HTML export |
| [docs/causalab-portability-audit.md](docs/causalab-portability-audit.md) | The CausaLab vocabulary alignment and its update procedure |
| [docs/perf-micro-design.md](docs/perf-micro-design.md) | Design of the perf microbenchmark (`scripts/perf.py`) |
| [docs/references.md](docs/references.md) | Related work by category, with its connection to nnbench |

The remaining files are records of one piece of work at one date. They describe the code as it
was then, so names of backends, files and flags in them may no longer exist.

| Record | Date | Content |
|---|---|---|
| [docs/large-model-handoff.md](docs/large-model-handoff.md) | 2026-10-22 | Handoff for the 72B comparison over two GPUs |
| [docs/delta-handoff.md](docs/delta-handoff.md) | 2026-09-30 | Handoff for the cross-system comparison on NCSA Delta through Apptainer |
| [docs/inference-benchmark-survey.md](docs/inference-benchmark-survey.md) | 2026-10-20 | Survey of how LLM serving benchmarks issue requests, used to design the load patterns |
| [docs/benchmark-contract-validation.md](docs/benchmark-contract-validation.md) | 2026-09-10 | Validation of the execution and artifact contract |
| [docs/runner-validation.md](docs/runner-validation.md) | 2026-09-08 | Validation of the independent backend runner |
| [docs/docker-validation.md](docs/docker-validation.md) | 2026-09-08 | Validation of the preceding shared-Compose runner |
| [docker/README.md](docker/README.md) | 2026-06-12, last changed 2026-10-13 | How to use the preceding shared-Compose runner |
| [docs/agents-and-the-primitive-model.md](docs/agents-and-the-primitive-model.md), [docs/drafts/](docs/drafts/) | 2026-06-11, 2026-06-12 | Design explorations folded into the design document |
| [docs/finding-batched-position-ids.md](docs/finding-batched-position-ids.md) | 2026-06-08 | nnsight batched traces on padded prompts of absolute-position models |

# Independent backend jobs

The main benchmark discovers **directory names** under `backends/`. Each directory has its own
`compose.yml`, Dockerfile (unless using a prebuilt image), and executable entrypoint. There is no
central backend registry or Compose override list. `scripts/bench.py` never selects an engine.

## Run

Requirements: Docker Engine, Docker Compose v2, NVIDIA Container Toolkit for the bundled GPU
backends, and host Python with CPU PyTorch for artifact scoring. Host nnsight/vLLM are not needed.

```bash
python scripts/bench.py list backends
python scripts/bench.py list specs
python scripts/bench.py build nnsight-hf nnsight-vllm

python scripts/bench.py run --spec logit_lens_gpt2 --data factual:32 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0

# Several experiments; one isolated container per experiment/backend pair.
python scripts/bench.py run --spec logit_lens_gpt2 steering_gpt2 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0

# Collect without a correctness reference. No hidden control runs.
python scripts/bench.py run --spec all --backends nnsight-hf --gpu 0

# Re-score saved artifacts; no Docker/model startup.
python scripts/bench.py score runs/REPLACE_WITH_RUN_ID
```

`--out` is a parent directory, not a reusable output filename. Each invocation creates a unique
UTC timestamp/UUID subdirectory and prints its location. `--timeout` limits each job (default
1800 seconds). `--seed` fixes worker random seeds. The selected `--reference` must be in
`--backends`. `--comparison equivalence` compares configurations rather than claiming correctness.
Neither the comparison axis nor reference is inferred from backend names.

Jobs run sequentially on the selected host GPU index/UUID. Only that GPU is exposed by the bundled
Compose configurations. Multiple timed jobs must not share an allocation; the runner is not a
cluster scheduler or a cross-process GPU reservation service.

The images pin nnsight and their library stacks. Repository code and the backend directory are
mounted read-only; output files are written as the invoking UID/GID. The model cache is a persistent
named volume (`isb-model-cache`, override with `ISB_MODEL_CACHE`). New caches download model files;
for an already populated cache, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` avoids network access.
Provide `HF_TOKEN` only when required. Credentials are not copied into execution records.

## Add another configuration

For an fp32 variant of the same vLLM implementation:

```bash
cp -R backends/nnsight-vllm backends/nnsight-vllm-fp32
# In the new compose.yml, change ISB_ENGINE_OPTIONS dtype to float32.
python scripts/bench.py run --spec logit_lens_gpt2 --data factual:2 \
  --backends nnsight-hf nnsight-vllm-fp32 --reference nnsight-hf --gpu 0
```

No Python registration or launcher edit is needed. A runtime-only variant can reuse the image. If
changing dependencies, choose a distinct image tag in the new Compose file before building it.
All backend-specific options live in that directory. The bundled `run.py` files translate existing
spec model-load hints and reject conflicts with configured options. Those compatibility hints are
retained in the Python specs; the host launcher does not interpret them.

A new inference engine needs actual method implementations, not just a Dockerfile. It can supply
its own executable, provided it satisfies the file contract below. The shared nnsight worker is
optional. A text-generation-only provider cannot claim support for inaccessible hidden states,
interventions, or gradients.

## Container contract (version 2)

Each Compose configuration defines a service called `runner`. Its entrypoint consumes `/job` and
writes `/output`. The host supplies these environment variables for Compose interpolation:

| Variable | Meaning |
|---|---|
| `ISB_JOB_DIR` | Absolute host directory mounted read-only at `/job` |
| `ISB_OUTPUT_DIR` | Absolute host directory mounted read/write at `/output` |
| `ISB_BACKEND` | Directory name, forwarded to the worker |
| `ISB_GPU` | Selected host GPU index/UUID; backend Compose determines device requirements |
| `ISB_UID`, `ISB_GID` | Invoking user, for output ownership |

Inputs are `experiment.json` and `inputs.jsonl`. The launcher resolves data once, records its exact
bytes/checksum, and hashes the resolved spec, seed and inputs into the experiment identity. Tuple
units/parameters are encoded as `{"$tuple": [...]}` so paired/labeled data round-trips without
becoming independent prompts. A job is a whole experiment, including dataset-level work such as
DAS training; the host does not shard its inputs.

The spec record is the Python dataclass as written: `regimes` (units replaced by their count),
tasks as `{label, params}`, the methodology's protocol template, and, for a methodology without
one, its `protocol_absence_reason`. All of it is covered by the experiment id. Readers support
both historical version-1 forms, which use `workloads` and `(params, label)` tasks:

- Files with an optional `description` retain their frozen protocol template, task parameters,
  and recorded coverage reason. Duplicate task descriptions must match the executable parameters.
- Files without a description retain unknown protocol coverage, including for a methodology
  that has a descriptor in today's code. Restoring them preserves their executable parameters.

Re-saving a historically undescribed spec as version 2 adds `protocol_source: "legacy"` to
preserve that distinction from an explicitly authored opt-out.

Migration keeps the original experiment id and bytes. These hashes identify runner inputs and
artifacts; CausaLab maintains its own document and artifact identities. Newly authored methods
remain subject to strict protocol validation; saved calls are checked against their frozen
descriptor at execution, where errors are recorded with the affected call.

Independent executables implement this versioned file contract. A pinned version-1-only reader
must reject version-2 jobs clearly; update the executable before submitting new jobs to it. The
bundled entrypoints use mounted repository code, so a reader update reaches them through the
source mount. Dependency changes still require rebuilding the images.

Built-in methods require protocol descriptors. A custom method can supply an `InterventionSpec`
directly or opt out with a nonempty `CellConfig.protocol_absence_reason`. Coverage is recorded
as `protocol_coverage` in the shared worker's provenance: `{"status": "described"}` or
`{"status": "undescribed", "reason": ...}`. Historically missing descriptions use
`{"status": "legacy", "reason": ...}`. Coverage describes metadata availability independently
of execution verdicts.

What is frozen: the experiment pins the spec and its inputs, and the launcher pins the executed
image and source commit. The shared worker binds each case's params to its cell's signature at
run time and records the effective params, split by the protocol, in `result.json` next to the
cell's timing, together with the baseline, effect-guard, and batched-reference call settings.
Reproduction uses the frozen experiment, source, and image together. Binding, metadata
description, and fresh invocation-parameter copies happen outside the measured interval.

The worker writes `result.pt` (the existing outputs/provenance dictionary) followed atomically by
`result.json` (completion, experiment/backend identity, input/tensor checksums, one execution status
for every requested cell, timings and provenance). See `isb/jobs/contract.py` and `worker.py` for
the precise fields. Tensor artifacts are trusted local Python/PyTorch artifacts, not an untrusted
network interchange format.

The shared worker also writes optional `auxiliary_calls` entries as
`{"key": [role, label], "record": {...}}`. Roles are `__baseline__`, `__effect__`, and
`batched_perprompt`; the second key is the regime for the first two and case label for the last.
These records preserve supporting-call settings and errors independently of task verdicts.
Independent workers may omit this field or supply an empty list.

Malformed input and spec-restoration failures also produce an atomic `result.json` with
`status: "failed"` and an error. Experiment identity is included only after its checksum has
been verified. These failures publish no successful tensor artifact and exit nonzero.

The launcher records actual image ID, source content checksum/commit, exit status and isolated
Compose project in `execution.json`, and captures `execution.log` and `cleanup.log`. Source changes
during a job invalidate it. Scoring also refuses mismatched source/model revisions/tokenizer
vocabularies; unresolved model identity cannot silently pass. Models are not automatically pinned
by the host: the worker records the resolved revision and comparisons require an exact match.

Supporting services may be declared in the same Compose file with health checks and `depends_on`.
Each job gets a unique project, which is cleaned on completion, timeout or interruption. Do not use
fixed container names, host networking, published ports, or external networks. Persistent model
volumes are retained; job cleanup never uses `down --volumes`.

## Results and exit codes

```text
runs/<run-id>/
  plan.json
  experiments/<experiment-id>/
    experiment.json
    inputs.jsonl
    <backend>/
      execution.json
      execution.log
      cleanup.log
      result.json
      result.pt
  report.json
```

Execution errors and benchmark findings are distinct. An individual method can produce `ERROR`
while the container completes its coverage report. A crash, timeout, missing/corrupted artifact,
invalid comparison, or unavailable reference makes the command exit nonzero. Add `--strict` to
also fail on unsuccessful cell verdicts. Without a reference, successful cells remain `RAN`.
Failures of one backend do not prevent the remaining jobs from collecting results.

Correctness scoring uses the reference's unpadded per-prompt results for batched candidates;
configuration equivalence compares matching workload regimes. There are no automatic precision
controls. Existing top-1/TV metrics are retained, with finite/shape checks and bounded vocabulary
padding alignment. Write-method effect guards must be present and strong for the applicable
interactive/generation reference, or the verdict is `INVALID_REFERENCE`.

The new launcher covers methodology specs. Separate micro/perf entrypoints and the old optional
serve tooling under `docker/` are legacy paths, not backends of this runner. Existing `.pt` files
remain usable by the legacy scorer and by the manager after an explicit `--import-legacy` step;
the new scorer requires the complete job directory. See [the result-site guide](../docs/result-site.md).

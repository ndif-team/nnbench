# Backend systems and configs

A **system** is a top-level directory under `backends/`: one interpretability system on one engine
(`nnsight-vllm`, `nnsight-hf`, `vllm-plain`, `vllm-lens`, `interp-engine-vllm`,
`transformer-lens-vllm`, `vllm-hook`). A **config** is an entry of the system's `configs.yml`; the
backend name is `<system>/<config>`, and a bare system name means `<system>/default`. There is no
central backend registry, no config directories, and `scripts/bench.py` never selects an engine.

```text
backends/<system>/
  Dockerfile      the system's image; versions are ARGs that the catalog's images set
  compose.yml     the runner service, parameterized by ISB_IMAGE, ISB_ENGINE_OPTIONS, ISB_GPUS, ISB_RANKS
  configs.yml     the catalog: images (Dockerfile build arguments) and configs (image, options, gpus, ranks)
  backend.py      create_backend(spec) -> the backend object; a `mode` option selects a realization
  cells.py        the system's workload implementation; the nnsight systems register theirs in
                  isb/methodologies/ and have none
  *.py            modules the two import: cells_opt.py (installed edits), nnsight04.py, nnsight05.py,
                  cells_nnsight04.py (adapters for the legacy nnsight releases)
```

## The catalog

```yaml
images:
  default: {}                                   # the Dockerfile's own ARG values
  0-8-1: {VLLM_IMAGE: vllm/vllm-openai:v0.27.1, NNSIGHT_REF: de7fec4d207fc8e0deacaa44fb68f8d0412506ef}
configs:
  default: {image: default, options: &base {dtype: bfloat16, gpu_memory_utilization: 0.85}}
  0-8-1: {image: 0-8-1, options: *base}
  0-8-1/async: {image: 0-8-1, options: {<<: *base, mode: async}}
  0-8-1/tp2: {image: 0-8-1, options: {<<: *base, tensor_parallel_size: 2}, gpus: 2}
```

Configs are named `<version>[/<mode>][/<flags>][/<tp>]` and every one is listed: nothing is derived
from a directory tree. A config on the `default` image has no version segment (`default`, `async`,
`fp32`): it runs the Dockerfile's own ARG values. `image` names an entry of `images`; the image tag is
`isb-<system>-<image>:local`. `options` is the JSON the worker receives as `ISB_ENGINE_OPTIONS` and
hands to `create_backend`; every config names it (`{}` for the engine's defaults). `gpus` (default
1) is how many host GPUs the job reserves, from `--gpu` on; `ranks` (default 1) is the number of
processes per job (`torchrun` ranks for HF tensor parallelism; every rank runs the job and rank 0
publishes it). `isb/jobs/layout.py` validates the file. For nnsight-vllm, `mode` is `sync` (the
default: nnsight's documented `LLM` path, one `tracer.invoke` per unit in one trace), `async`
(`AsyncLLM`, one request per unit) or `opt` (installed edits and plain requests, `cells_opt.py`).

## Run

Requirements: Docker Engine, Docker Compose v2, the NVIDIA Container Toolkit, and host Python with
CPU PyTorch for scoring. Host nnsight and vLLM are not needed. Without Docker,
`ISB_LAUNCHER=apptainer` builds the same Dockerfiles into `.sif` images and runs the same service
(`isb/jobs/apptainer.py`; see the top-level README).

```bash
python scripts/bench.py list backends
python scripts/bench.py list workloads   # also: models, suites
python scripts/bench.py build nnsight-hf/0-8-1 nnsight-vllm/0-8-1 nnsight-vllm/0-8-1/async   # two images

python scripts/bench.py run --workload logit_lens --model gpt2 --data factual:32 \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 --reference nnsight-hf/0-8-1 --gpu 0

# Several workloads; one isolated container per experiment/backend pair.
python scripts/bench.py run --workload logit_lens steering --model gpt2 \
  --backends nnsight-hf/default nnsight-vllm/default --reference nnsight-hf/default --gpu 0

# Collect without a correctness reference. No hidden control runs.
python scripts/bench.py run --workload logit_lens --model gpt2 --backends nnsight-hf/default --gpu 0

# A suite (isb/suites.py): its workloads x models, backends and reference.
python scripts/bench.py run --suite smoke --gpu 0

# Re-score saved artifacts; no Docker/model startup.
python scripts/bench.py score runs/REPLACE_WITH_RUN_ID
```

`build` builds each distinct (system, image) once. `run` starts one ephemeral container per
(experiment, backend) from the config's image, with the config's options as `ISB_ENGINE_OPTIONS`
and its GPUs as `ISB_GPUS`: `--gpu N` and the next `gpus - 1` indices, so `--gpu` must be an index
for a config reserving several (`*/tp2`, `*/gpu2`). Jobs run one after another; multiple timed jobs
must not share an allocation. `--out` is a parent directory, not a reusable output filename: each
invocation creates a unique UTC timestamp/UUID subdirectory and prints its location. `--timeout`
limits each job (default 1800 seconds). `--seed` fixes worker random seeds. The selected
`--reference` must be in `--backends`. `--comparison equivalence` compares configurations rather
than claiming correctness. Neither the comparison axis nor the reference is inferred from names.

Repository code, including `backends/`, is mounted read-only at `/workspace`; output files are
written as the invoking UID/GID. The model cache is a persistent named volume (`isb-model-cache`,
override with `ISB_MODEL_CACHE`). New caches download model files; for a populated cache,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` avoids network access. Provide `HF_TOKEN` only when
required. Credentials are not copied into execution records.

## Add a configuration

Add an entry to the system's `configs.yml`; no file, directory or Python registration is needed:

```yaml
  0-8-1/fp32: {image: 0-8-1, options: {<<: *base, dtype: float32}}
```

A new version is a new `images` entry (its ARG values) plus the configs that use it. A realization
that needs different code is a `mode` option that the system's `backend.py` dispatches on, as
nnsight-vllm's `async` and `opt` do; it never becomes a directory. The bundled `backend.py` files
translate the model row's load options (`hf_kwargs`, `vllm_kwargs` in `isb/models.py`, frozen into
the experiment) and reject conflicts with configured options; the host launcher does not interpret
them.

A new system is a new directory with the files above and actual method implementations, not just a
Dockerfile. It can supply its own executable, provided it satisfies the file contract below. A
text-generation-only provider cannot claim support for inaccessible hidden states, interventions,
or gradients. Write its cells by the procedure in [writing workloads](../docs/writing-workloads.md):
every cell takes `units: list[Unit]` and processes every unit of the call in the system's
documented batch or concurrent form, and declares a workload the system cannot express by raising
`Unsupported` with the missing capability. Stay inside the system's public interface (§4a there): a
faster form that uses its internals is reported as headroom, not as its score.

## Container contract (version 2)

Each system's `compose.yml` defines a service called `runner`. Its entrypoint consumes `/job` and
writes `/output`. The host supplies these environment variables for Compose interpolation:

| Variable | Meaning |
|---|---|
| `ISB_JOB_DIR` | Absolute host directory mounted read-only at `/job` |
| `ISB_OUTPUT_DIR` | Absolute host directory mounted read/write at `/output` |
| `ISB_BACKEND` | Config name `<system>/<config>`, forwarded to the worker |
| `ISB_IMAGE` | The config's image tag, `isb-<system>-<image>:local` |
| `ISB_ENGINE_OPTIONS` | The config's options as JSON, handed to `create_backend` |
| `ISB_GPUS` | Comma-separated host GPU indices the job reserves |
| `ISB_RANKS` | Processes per job (`torchrun` ranks when above 1) |
| `ISB_UID`, `ISB_GID` | Invoking user, for output ownership |

Inputs are `experiment.json` and `inputs.jsonl`. The launcher resolves data once, records its exact
bytes/checksum, and hashes the resolved spec, seed and inputs into the experiment identity. Tuple
units/parameters are encoded as `{"$tuple": [...]}` so paired/labeled data round-trips without
becoming independent prompts. A job is a whole experiment, including dataset-level work such as
DAS training; the host does not shard its inputs.

The spec record is the Python dataclass as written: `regimes` (units replaced by their count),
tasks as `{label, params}`, the methodology's protocol template, and, for a methodology without
one, its `protocol_absence_reason`. All of it is covered by the experiment id. Readers accept
version 2. Historical version-1 jobs must be rerun. Saved descriptors and absence reasons stay
frozen when a current job is restored.

These hashes identify runner inputs and artifacts; CausaLab maintains its own document and
artifact identities. Newly authored methods remain subject to strict protocol validation; saved
calls are checked against their frozen descriptor at execution, where errors are recorded with the
affected call.

Independent executables implement this versioned file contract. A pinned version-1-only reader
must reject version-2 jobs clearly; update the executable before submitting new jobs to it. The
bundled entrypoints use mounted repository code, so a reader update reaches them through the
source mount. Dependency changes still require rebuilding the images.

Built-in methods require protocol descriptors. A custom method can supply an `InterventionSpec`
directly or opt out with a nonempty `CellConfig.protocol_absence_reason`. Coverage is recorded
as `protocol_coverage` in the shared worker's provenance: `{"status": "described"}` or
`{"status": "undescribed", "reason": ...}`. Coverage describes metadata availability independently
of execution verdicts.

What is frozen: the experiment pins the spec and its inputs, and the launcher pins the executed
image and source commit. The shared worker binds each case's params to its cell's signature at
run time and records the effective params, split by the protocol, in `result.json` next to the
cell's timing, together with the baseline, effect-guard, and per-unit reference call settings.
Bound automatic selectors stay explicit (for example `params.residual = null`); `resolved_params`
records choices reported by the cell at its existing resolution site. Unreported choices have no
entry in that map. Reproduction uses the frozen experiment, source, and image together. Binding,
metadata description, and fresh invocation-parameter copies happen outside the measured interval.

The worker writes `result.pt` (the outputs/provenance dictionary) followed atomically by
`result.json` (completion, experiment/backend identity, input/tensor checksums, one execution status
for every requested cell, timings and provenance). See `isb/jobs/contract.py` and `worker.py` for
the precise fields. Tensor artifacts are trusted local Python/PyTorch artifacts, not an untrusted
network interchange format.

The shared worker also writes optional `auxiliary_calls` entries as
`{"key": [role, label], "record": {...}}`. Roles are `__baseline__`, `__effect__`, `__vanilla__`,
`batched_perprompt` and `offline_perprompt`; the second key is the regime for the first three and
the case label for the last two. `__vanilla__` times the backend's own request with no intervention
attached (the backend's `plain` method over the same units, in the same call form and timing loop
as the cells), or records why it is unavailable. These records preserve supporting-call settings
and errors independently of task verdicts. Independent workers may omit this field or supply an
empty list.

Malformed input and spec-restoration failures also produce an atomic `result.json` with
`status: "failed"` and an error. Experiment identity is included only after its checksum has
been verified. These failures publish no successful tensor artifact and exit nonzero.

The launcher records the actual image ID, source content checksum/commit, exit status and isolated
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
    <system>/<config>/          one directory per backend name (0-8-1/opt/taps nests as written)
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

Correctness scoring uses the reference's unpadded per-unit results for multi-unit candidates;
configuration equivalence compares matching workload regimes. There are no automatic precision
controls. Write-method effect guards must be present and strong for the applicable
interactive/generation reference, or the verdict is `INVALID_REFERENCE`.

The launcher covers methodology specs. Separate micro/perf entrypoints and the old serve tooling
under `docker/` are legacy paths, not backends of this runner. Current schema-3 `.pt` files are
usable by the standalone scorer and by the manager after an explicit `--import-legacy` step; the
new scorer requires the complete job directory. See [the result-site guide](../docs/result-site.md).

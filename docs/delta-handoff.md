# Handoff: cross-system comparison on Delta

Goal: run the cross-system comparison (design.md §12.14) on an uncontended A100, so its latencies
are usable. The shared host where it was developed had other workloads take the GPU mid-run.
Correctness results from that host are already recorded in `docs/findings.md`; Delta's job is
clean timing, plus confirming the same correctness verdicts.

Branch `feat/cross-system-comparison`; the Apptainer launcher needs the commit that adds `isb/jobs/apptainer.py` or later.

## What this branch adds

- Five comparison specs on Qwen/Qwen2.5-1.5B-Instruct: `cmp_logit_lens`, `cmp_steering`,
  `cmp_gen_steering`, `cmp_activation_patching`, `cmp_ablation` (`isb/specs/comparison.py`).
- Backends, one Docker Compose directory each under `backends/`:

| backend | system | vLLM | engine mode |
|---|---|---|---|
| `nnsight-hf` | nnsight on transformers 5.12.1 (the reference) | — | eager PyTorch |
| `nnsight-vllm` | nnsight 0.8.0rc1 (260c555) | 0.19.1 | eager |
| `vllm-lens` | vLLM-Lens 1.2.1 | 0.19.1 | eager (forced by the plugin) |
| `interp-engine` | interp-engine 1.12.0 | 0.28.0 | eager |
| `transformer-lens` | TransformerLens 4.0.0 `RemoteBridge.boot_vllm` | 0.20.2 | torch.compile + CUDA graphs |
| `vllm-plain-0-19-1`, `-0-20-2`, `-0-28-0` | plain vLLM, nothing installed | each | torch.compile + CUDA graphs |

- Two denominators per timed cell: `overhead_vs_vanilla` (the same system with no intervention
  attached, timed in the same job) and `overhead_vs_plain_vllm` (plain vLLM at the same version).
- fp32 variants `nnsight-hf-fp32`, `nnsight-vllm-fp32`, `transformer-lens-fp32` for separating
  precision from intervention differences.

## Step 0: Delta has no Docker; use the Apptainer launcher

Checked 2026-09-30 (job 22588849, gpuA100x4, node gpua007): no Docker Engine, Compose or podman;
Apptainer 1.5.3 on compute nodes, and `apptainer exec --nv` sees the GPU (A100-SXM4-40GB, driver
595.71.05). The runner therefore uses its Apptainer launch path (`isb/jobs/apptainer.py`). It reads
the same `compose.yml` and Dockerfile as the Docker path: a build translates the Dockerfile's
ARG/FROM/RUN lines into an Apptainer definition, and a run maps the runner service onto
`apptainer exec --nv --cleanenv --no-home` with the same `/job`, `/output`, `/workspace`,
`/backend` and `/models` mounts.

Set these in every shell (the login node for builds, the allocation for runs):

```bash
cd /work/nvme/bdnh/zwang83/nnbench          # or wherever the checkout lives on /work
export ISB_LAUNCHER=apptainer               # every bench.py build/run below uses Apptainer
export ISB_APPTAINER_DIR=$PWD/.apptainer    # .sif images, definitions, model cache (not tracked)
export APPTAINER_CACHEDIR=/work/nvme/bdnh/zwang83/.apptainer-cache   # layer cache, off $HOME
export APPTAINER_TMPDIR=/tmp                # build scratch; node-local and large
srun --account=bdnh-delta-gpu --partition=gpuA100x4 --gpus=1 --cpus-per-task=16 --mem=64G \
     --time=03:00:00 --pty bash             # for runs; builds can also run here
```

Builds run the Dockerfiles' `RUN` steps (apt-get, pip) as root, so they use `--fakeroot` by
default. Check it works with the cheapest image first (plain vLLM 0.19.1 adds only a `mkdir`):

```bash
python scripts/bench.py build vllm-plain-0-19-1
ls -la .apptainer/images/
```

If `--fakeroot` is refused, try `export ISB_APPTAINER_BUILD_FLAGS=""` (an unprivileged build) and
report which worked. Building the vLLM images needs memory and time; run builds inside an
allocation if the login node kills them.

Notes:
- Disk: about 50 GB of `.sif` images under `ISB_APPTAINER_DIR`, plus the layer cache.
- The model cache is `$ISB_APPTAINER_DIR/volumes/isb-model-cache`, mounted at `/models`
  (`HF_HUB_CACHE=/models/hub`). Qwen2.5-1.5B-Instruct is public; no `HF_TOKEN` is needed. If
  compute nodes cannot reach the Hub, prefetch on the login node:
  `HF_HUB_CACHE=$ISB_APPTAINER_DIR/volumes/isb-model-cache/hub huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct`.
- The GPUs are A100-SXM4-40GB. Every vLLM backend runs at `gpu_memory_utilization` 0.85 (34 GB
  here; at 0.9 TransformerLens's full-vocabulary sampler warmup ran out of memory), the same in every `compose.yml`. The earlier 0.2 was for the shared development host; on
  an exclusive Delta allocation one job owns the GPU. Any change to the fraction must be made for
  every backend alike and committed before the run. The `cmp_*` specs have no batched regime, so
  nnsight-vllm never loads its co-resident sync twin (`isb/sweep/execute.py`, `_load_sync_twin`),
  which would not fit beside a 0.9 engine.
- The vLLM 0.28.0 image ships CUDA 13.0 torch; driver 595 supports it natively.
- Inside an allocation the GPU is index 0: pass `--gpu 0`.
- Run on a GPU no one else uses for the duration. Every job records the GPU's used memory before
  and after (`execution.json`, field `gpu_release`); a nonzero `baseline_mib` means something else
  was on the GPU.

## Step 1: no-GPU tests

```bash
git fetch origin && git checkout feat/cross-system-comparison
python -m pytest tests/ -q          # expect 509 passed, 3 skipped (host Python with CPU torch and PyYAML)
```

## Step 2: smoke tests, smallest first

Each run prints one line per cell and writes `runs/<out>/<timestamp>/report.json`. Build only what
each step needs; later steps reuse the images.

**2a. nnsight on HF and vLLM, one spec, four prompts**

```bash
python scripts/bench.py build nnsight-hf nnsight-vllm
python scripts/bench.py run --spec cmp_steering --data counterfact:4 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0 --out runs/delta-smoke
```

Expect every `nnsight-vllm` row `SUPPORTED` with top-1 1.00, TV 0.000.

**2b. vLLM-Lens and plain vLLM 0.19.1 (same base image)**

```bash
python scripts/bench.py build vllm-lens vllm-plain-0-19-1
python scripts/bench.py run --spec cmp_steering --data counterfact:4 \
  --backends nnsight-hf vllm-lens vllm-plain-0-19-1 --reference nnsight-hf --gpu 0 --out runs/delta-smoke
```

Expect `vllm-lens` `SUPPORTED` on all four rows and `vllm-plain-0-19-1` `UNSUPPORTED` on all four
(plain vLLM has no intervention layer; it only contributes timing). In `report.json`, the
`vllm-lens` rows should carry `overhead_vs_plain_vllm`.

**2c. interp-engine and TransformerLens with their plain vLLM versions**

```bash
python scripts/bench.py build interp-engine transformer-lens vllm-plain-0-20-2 vllm-plain-0-28-0
python scripts/bench.py run --spec cmp_steering cmp_ablation --data counterfact:4 \
  --backends nnsight-hf interp-engine transformer-lens vllm-plain-0-20-2 vllm-plain-0-28-0 \
  --reference nnsight-hf --gpu 0 --out runs/delta-smoke
```

Expect, for `cmp_steering`: `interp-engine` `SUPPORTED` on all four rows; `transformer-lens`
`SUPPORTED` on the two mean-norm rows and `UNSUPPORTED` on the two "scaled by each token's norm"
rows. For `cmp_ablation`: `interp-engine` `UNSUPPORTED`, `transformer-lens` `NUMERICAL_MISMATCH`
(bf16 precision; it matches HF at fp32).

**2d. Patching, which uses paired data**

```bash
python scripts/bench.py run --spec cmp_activation_patching --data mib/ioi:4 \
  --backends nnsight-hf nnsight-vllm vllm-lens interp-engine transformer-lens \
  --reference nnsight-hf --gpu 0 --out runs/delta-smoke
```

## Step 3: the full comparison

About 40 jobs, roughly 75 minutes on one A100.

```bash
python scripts/bench.py run \
  --spec cmp_logit_lens cmp_steering cmp_gen_steering cmp_activation_patching cmp_ablation \
  --backends nnsight-hf nnsight-vllm vllm-lens interp-engine transformer-lens \
             vllm-plain-0-19-1 vllm-plain-0-20-2 vllm-plain-0-28-0 \
  --reference nnsight-hf --gpu 0 --timeout 3600 --out runs/comparison-delta
```

Optional precision check:

```bash
python scripts/bench.py build nnsight-hf-fp32 nnsight-vllm-fp32 transformer-lens-fp32
python scripts/bench.py run --spec cmp_ablation \
  --backends nnsight-hf-fp32 nnsight-vllm-fp32 transformer-lens-fp32 \
  --reference nnsight-hf-fp32 --gpu 0 --out runs/comparison-delta
```

## Expected verdicts (from the development host)

Top-1 agreement / TV against `nnsight-hf`. If Delta differs, report it rather than changing cells.

| workload | nnsight-vllm | vllm-lens | interp-engine | transformer-lens |
|---|---|---|---|---|
| logit lens, every layer | 0.82 / 0.136 | same, bitwise equal to nnsight-vllm | 0.82 / 0.136 | 0.82 / 0.136 |
| steering, mean or token norm | 1.00 / 0.000 | 1.00 / 0.000 | 1.00 / 0.000 | mean: 1.00 / 0.000; token: UNSUPPORTED |
| every-step steering, mean norm | 1.00 / 0.000 | 1.00 / 0.000 | UNSUPPORTED | UNSUPPORTED |
| every-step steering, token norm | 1.00 / 0.000 | 1.00 / 0.000 | 1.00 / 0.000 | UNSUPPORTED |
| patching, every position | 1.00 / 0.09 | 1.00 / 0.09 | UNSUPPORTED | UNSUPPORTED |
| patching, last token | 0.88 / 0.07–0.10 | 0.88–0.94 / 0.07–0.09 | 0.88 / 0.07–0.10 | 0.88 / 0.07–0.10 |
| ablation, MLP / attention, layer 1 | 0.78 / 0.19; 0.50 / 0.43 | UNSUPPORTED | UNSUPPORTED | 0.75 / 0.20; 0.56 / 0.43 |

The logit-lens and patching gaps are the vLLM engine's distance from HF (all four vLLM systems land
there); the ablation gap closes at fp32. Every `UNSUPPORTED` row carries its reason, citing the
system's source, in the `unsupported` field of the cell.

Uncontended latency for a single forward with nothing attached, measured before contention began:
plain vLLM 0.19.1 8.4 ms, 0.20.2 11.4 ms, 0.28.0 11.5 ms; nnsight-vllm 20.8 ms; vLLM-Lens 21.6 ms;
interp-engine 21.7 ms; TransformerLens 11.6 ms.

## Reading and returning results

```bash
python scripts/manager.py --dir runs/comparison-delta --export comparison-delta.html
tar czf comparison-delta.tgz runs/comparison-delta        # runs/ is git-ignored
```

Per cell in `report.json`: `state`, `metrics` (top1_agree, tv, max_abs), `median_latency_ms`,
`overhead_vs_vanilla`, `overhead_vs_plain_vllm`, and `unsupported` for declared gaps. Per job in
`execution.json`: `status`, `image_id`, `source`, `gpu_release`. Send back the tarball and the
HTML export.

## Known issues

- A vLLM job that starts while the previous job's GPU memory is still draining can fail vLLM's
  memory-profiling assertion ("Error in memory profiling") or find no memory for the KV cache. The
  runner now waits up to 120 s for the memory to return to its pre-job level. If it still
  happens, rerun the failed job; it is not a cell failure.
- "Free memory on device ... less than desired GPU memory utilization" means another process holds
  the GPU. The result is unusable for timing.
- vLLM's usage-statistics thread logs `PermissionError: '/.config'` in every vLLM container. It is
  harmless.

## Rules for working on this branch

- Do not edit `isb/` or `backends/` while a run is active: the job records a source hash, and a
  change mid-run fails the job.
- New or changed cells follow `docs/writing-workloads.md`: read the system's docs at the pinned
  version, check what each read and write site denotes in source, make the documented form the
  default, test the denotation.
- A mismatch is reported, not patched around. A divergence that only appears under tensor or
  pipeline parallelism is a finding to classify (CLAUDE.md).

## Apptainer launcher details

- Launch method: `ISB_LAUNCHER=apptainer` or `--launcher apptainer` on `build` and `run`; recorded
  as `launcher` in `plan.json` and each `execution.json`. It is a runner setting, never chosen per
  backend.
- Image identity: the `.sif` file's sha256 (`image_id` in `execution.json`), cached beside it.
- A Dockerfile instruction other than ARG, FROM or RUN, a multi-stage build, or a compose file with
  supporting services is refused with a message; extend `isb/jobs/apptainer.py` rather than
  working around it. Never run the backends from conda envs instead: the comparison depends on
  each system's pinned image.
- Environment reaches the container as `APPTAINERENV_*` variables (so JSON values with commas pass
  intact); the GPU is selected with `CUDA_VISIBLE_DEVICES`. A timeout kills the container's whole
  process group.
- The runner's GPU-memory wait uses the host's `nvidia-smi`, available on Delta GPU nodes.
- Not yet run on a real Apptainer host; the CPU tests cover the translation, the command mapping
  and a job lifecycle. The first build and the first smoke run are the real checks.

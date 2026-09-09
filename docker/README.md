# Backend containers

> **Legacy tooling.** The main benchmark no longer reads this shared Compose file or accepts
> the old `--compose-file`/engine-name CLI shown below. Use
> [`backends/README.md`](../backends/README.md) and
> `python scripts/bench.py build nnsight-hf nnsight-vllm` for the independent backend runner.
> The material below documents the previous setup; `run_vm.sh` now invokes the legacy standalone
> executor directly and is outside the new HF/local-vLLM validation.

Every benchmark backend runs in its own Docker Compose service. Docker owns the Python and package
versions; `scripts/bench.py` only chooses a service and passes the normal `scripts/execute.py`
arguments. The container writes its run file through a host-directory mount at `/outputs`, then the
host scorer reads the same file.

The services are deliberately direct:

| benchmark backend | Compose service | image |
|---|---|---|
| `hf` | `hf` | `Dockerfile.hf` |
| `vllm_async` | `vllm_async` | `Dockerfile` |
| `vllm_sync` | `vllm_sync` | `Dockerfile` |
| `vllm_serve` | `vllm_serve` | `Dockerfile` |

The vLLM services share one image because their package setup is identical; each service's
entrypoint supplies its engine and execution mode to `execute.py`. Each Dockerfile pins the base commit of the corresponding
nnsight development checkout; the HF image also pins its transformers stack and the vLLM image
inherits its version from the pinned vLLM base image. Uncommitted changes in those external
checkouts are intentionally not copied into these images.

## Build and run

From the repository root:

```bash
docker compose -f docker/docker-compose.yml build hf vllm_async

# Small end-to-end run: HF reference, vLLM candidate, fp32 vLLM control, then CPU scoring.
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec logit_lens_gpt2 \
  --data factual:2 --out runs/docker-smoke

CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec steering_gpt2
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec all --backends hf
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec steering_gpt2 \
  --backends hf vllm_sync --out results/runs
```

`bench.py` expands the host output path and mounts it into each one-shot container. No Conda path or
backend-specific interpreter argument is involved. Model downloads are retained in the Compose
`model-cache` volume.

Requirements: Docker Engine with NVIDIA Container Toolkit, Compose v2.30+ (the `gpus` service
setting), and a host Python with PyTorch for scoring. A CPU PyTorch installation is sufficient on
the host (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`). Backend
packages, including nnsight and vLLM, are installed only inside the images. Use a free GPU for
performance measurements; the smoke command checks execution and equivalence.

The logit-lens suite includes deliberately unsupported cells. On the pinned vLLM stack the
interactive `unembed=weight` cell works; direct `lm_head.forward` and multi-prompt batched saves
produce `ERROR` rows. These rows are benchmark findings, while a container startup/model-load
failure makes `bench.py` exit unsuccessfully.

## Plug in a different container

`--backends` takes Compose service names. The launcher passes `--spec`, `--data`, `--name`, and
`--out /outputs` to the service's entrypoint. That entrypoint owns the implementation and writes
the usual run file. No additional manifest or Python service registration is required.

`compose.variant.example.yml` adds an fp32 vLLM candidate by extending the existing service:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py \
  --compose-file docker/docker-compose.yml \
  --compose-file docker/compose.variant.example.yml \
  --backends hf vllm_candidate --spec logit_lens_gpt2 --data factual:2 \
  --out runs/docker-variant
```

Change `image`, `build`, environment variables, mounts, or the entrypoint in Compose to supply
another setup. The bundled names keep their historical run tags (`hf`, `vllm`, etc.); custom
services use their own names. HF is the reference when selected; otherwise the first service is.
Automatic precision controls apply to the bundled vLLM services. For a custom candidate, choose
its precision in Compose and supply a separately executed control to `scripts/score.py` if needed.
Run files record the installed nnsight Git commit using pip's installation metadata.

## Optional serve backend (outside the HF/vLLM validation)

The Compose file also contains an optional `server` service. `docker/run_vm.sh` groups specs by model,
starts that server, and lets `bench.py` launch the GPU-less `vllm_serve` runner against it:

```bash
GPU=0 docker/run_vm.sh
GPU=0 docker/run_vm.sh steering_gpt2 ablation_gpt2
```

Reference files must already exist in `results/runs`; stock them with normal containerized runs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec all --backends hf --out results/runs
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec all --ctl-only --out results/runs
```

## Security

The bundled serve endpoint accepts serialized nnsight execution requests. It binds inside the
private Compose network and is intentionally not published to a host port. Do not add a public port
mapping for it.

# Docker execution validation — 2026-09-08

> Historical validation of the preceding shared-Compose runner. The current name-based runner
> and independent backend directories are validated in [runner-validation.md](runner-validation.md).

The HF reference, vLLM candidate, and fp32 vLLM control built and ran through
`scripts/bench.py` using the Compose services. Tests ran on an NVIDIA A100 80GB PCIe (GPU 5).
This was a functional smoke test on a shared host, not a performance measurement.

## Reproduce

```bash
docker compose -f docker/docker-compose.yml build hf vllm_async
CUDA_VISIBLE_DEVICES=0 python scripts/bench.py --spec logit_lens_gpt2 \
  --data factual:2 --out runs/docker-smoke
```

The validated run used `CUDA_VISIBLE_DEVICES=5`, `COMPOSE_PROJECT_NAME=nnbench-docker-test`,
and output directory `runs/docker-validated-XGvX5N`. Its command exited with status 0.
The directory contains `logit-lens.log` and the three self-contained `.pt` run files.

## Installed stacks

| Component | HF | vLLM |
|---|---|---|
| nnsight commit | `944c8056382477662f153a0fcc52487d81355281` | `5166eb12ceb3b78091f7a8673f0cb47fd453dcb9` |
| PyTorch | `2.11.0+cu128` | `2.10.0+cu129` |
| transformers | `5.12.1` | `5.5.4` |
| vLLM | — | `0.19.1` |
| built image ID | `sha256:910093b63f3d9aa42cf583027abf05250ffbd8cdd43732c1ba840d945d045424` | `sha256:12d4d469d521ca8b8dfb132b2496902bc07b3fa3c09bdba17b65e576506875e2` |

The source commits are installed from Git, without external development mounts or uncommitted
patches. The full commit is retained in each run's provenance via pip's installation metadata.
The images use the dependency versions above; these differ from the old local Conda environments.

## Measured results

Both comparisons below use the interactive `unembed=weight` logit-lens cell, over two factual
prompts and all 12 GPT-2 layers, scored against HF:

| Candidate | State | Top-1 agreement | Mean TV |
|---|---|---|---|
| vLLM bf16 | `SUPPORTED` | 1.0 | 0.0200627875 |
| vLLM fp32 control | `SUPPORTED` | 1.0 | 0.0005092705 |
| Compose-only `vllm_candidate` (fp32) | `SUPPORTED` | 1.0 | 0.0005092705 |

The additional candidate ran with the following command and exited with status 0:

```bash
CUDA_VISIBLE_DEVICES=5 COMPOSE_PROJECT_NAME=nnbench-docker-test python scripts/bench.py \
  --compose-file docker/docker-compose.yml \
  --compose-file docker/compose.variant.example.yml \
  --spec logit_lens_gpt2 --data factual:2 --backends vllm_candidate --score-vs hf \
  --out runs/docker-validated-XGvX5N
```

Its provenance confirms `engine.params.dtype=float32`, supplied entirely by Compose. Its log is
`custom-candidate.log`; its artifact is `logit_lens_gpt2@factual-vllm_candidate.pt`. The standard
run and this additional candidate leave no running containers in the test Compose project.

All four HF cells executed. Three vLLM cells record `ERROR`: the two direct LM-head calls are
guarded by vLLM, and the batched weight cell receives no saved outputs from this nnsight stack's
multi-prompt path. These are recorded coverage findings; cells were not removed or rewritten to
make the map pass. The working cell still produces finite outputs, timings, and a scored verdict.

## Checks and scope

- 182 no-GPU tests passed, including Compose overrides, Git-wheel provenance, and async shutdown.
- Compose configuration and shell syntax checks passed.
- The HF image passes `pip check`. The vLLM base image has an unrelated optional desktop-package
  warning (`pygobject` lacks `pycairo`); its CUDA engine and benchmark executed successfully.
- Async engine tasks are drained before loop closure; the final runs have no pending-task or
  closed-loop shutdown errors.
- HF and local async vLLM were GPU-tested. Serve, multi-GPU execution, and the separate micro/perf
  launchers are outside this validation.

See [the Docker setup](../docker/README.md) for the Compose-only candidate example.

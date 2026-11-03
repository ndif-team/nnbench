# Executable Delta Handoff: Same-Family Scaling & Large-Model Evaluation on NCSA Delta (`gpuH200x8`)

> **Target Executor:** Autonomous Coding & Evaluation Agent (Claude Opus on NCSA Delta) + Reviewer Subagents
> **Execution Status:** **ON HOLD / OPTIONAL POST-`HAKONE` RESERVE** — We are executing the entire core evaluation (`Waves 1–3`: `7B`, `8B`, `14B`, `32B`, `70B`) on `hakone` (`8× A100-SXM4-80GB`) first without `H200`. Only run this Delta handoff if we decide after reviewing the `hakone` results that we want `2× H200` `Qwen2.5-72B` (`TP=2`) or `8× H200` `Llama-3.1-405B` (`TP=8`).
> **Delta Checkout:** `/work/nvme/bdnh/zwang83/nnbench`
> **Target Branch & Commit:** `refactor/consolidated-architecture` (`d0ed4b9bfac87cefa2bf01c44650176c3f06a8d6`, `source_sha256`: `d3d100a1cd3effedbd8390330626b1bfdbd1b5aeb5d8ab38dd59cea240f1c31d`)
> **Launcher:** `ISB_LAUNCHER=apptainer` (`isb/jobs/apptainer.py`)
> **Target Partition:** `gpuH200x8` (`NVIDIA H200-141GB`, always `--exclude=gpue01` which is reserved for NDIF)

---

## 1. Purpose & Scope: Why These Runs Are on Delta (`gpuH200x8`)

Stages 1–5 for `<= 70B` capability, cross-system latency, and version evolution are already running on `hakone` (`8× A100-SXM4-80GB`). However:
1. **`Qwen2.5-72B-Instruct` (`145 GB` bf16) at `TP=2` (`2 GPUs`)** requires `72.5 GB/GPU` of weights alone, which exceeds `hakone`'s `80 GB` A100s at `gpu_memory_utilization=0.85` (`68 GB` budget) and requires **`2× H200-141GB`** (`~68 GB/GPU` free headroom).
2. **Fair Same-Family Scaling Baselines (`7B → 14B → 72B`):** Because `72B` runs on `H200`, **its smaller same-family counterparts (`Qwen2.5-7B-Instruct` on `1× H200` and `Qwen2.5-14B-Instruct` on `1× H200` & `2× H200`) must also run on `H200`** so the scaling curves and regime tables (`tab:eval-regimes` in `sections/05-evaluation.tex`: `7B` vs. `72B` across `Offline` and `Concurrent` regimes, and `opt7b` vs. `optlarge` interactive realization headroom) compare models on **identical H200 hardware** rather than mixing `A100` (`2.0 TB/s` HBM2e) with `H200` (`4.8 TB/s` HBM3e).
3. **Zero Cross-Job Contention via Slurm:** All jobs below are submitted via `sbatch` to `gpuH200x8`, so Slurm schedules them across dedicated H200 GPUs (`1 + 2 + 4 = 7 GPUs`, fitting on a single `8× H200` node `gpue02`/`gpue03`) concurrently with hardware-enforced cgroup isolation.

---

## 2. Step 0 — Sync Codebase & Verify `source_identity()` Lock on Delta

On the Delta login node:
```bash
cd /work/nvme/bdnh/zwang83/nnbench
git fetch origin
git checkout d0ed4b9bfac87cefa2bf01c44650176c3f06a8d6

# Verify exact source_identity() lock matching hakone:
# commit: d0ed4b9bfac87cefa2bf01c44650176c3f06a8d6
# source_sha256: d3d100a1cd3effedbd8390330626b1bfdbd1b5aeb5d8ab38dd59cea240f1c31d
/work/nvme/bdnh/zwang83/envs/isb-host/bin/python -c "
from isb.jobs.local import source_identity
ident = source_identity()
print(ident)
assert ident['commit'] == 'd0ed4b9bfac87cefa2bf01c44650176c3f06a8d6'
assert ident['source_sha256'] == 'd3d100a1cd3effedbd8390330626b1bfdbd1b5aeb5d8ab38dd59cea240f1c31d'
"

# Run the no-GPU test suite (must pass 618 tests, including all 10 test_apptainer.py tests)
/work/nvme/bdnh/zwang83/envs/isb-host/bin/python -m pytest tests/ -q
```

> **CRITICAL RULE:** Do **not** modify any file under `isb/` or `backends/` before or during the `Qwen2.5` (`7B → 14B → 72B`) and `Llama-3.3-70B` runs so `source.commit` and `source.source_sha256` remain identical to `hakone` (`d0ed4b9...` / `d3d100a1...`). (Note: `slurm/*`, `scripts/*`, and `docs/*` are outside `source_identity()` and safe to update.)

---

## 3. Step 1 — Build Apptainer `.sif` Images via `slurm/delta-build.slurm`

The scaling and realization experiments only need **4 Apptainer `.sif` images** (`nnsight 0.8.1` + `vLLM 0.27.1` / `0.28.0` stack):
1. `isb-nnsight-hf-0-8-1_local.sif` (covers `nnsight-hf/0-8-1`, `nnsight-hf/0-8-1/gpu2`, `nnsight-hf/0-8-1/gpu4`)
2. `isb-nnsight-hf-0-8-1-tf5-19_local.sif` (covers `nnsight-hf/0-8-1/tp2`, `nnsight-hf/0-8-1/tp4`)
3. `isb-nnsight-vllm-0-8-1_local.sif` (covers `nnsight-vllm/0-8-1`, `async`, `opt`, `opt/taps`, and all their `tp2`/`tp4` variants!)
4. `isb-vllm-plain-0-27-1_local.sif` (covers `vllm-plain/0-27-1`, `eager`, `tp2`, `eager/tp2`, `tp4`, `eager/tp4`)

Submit 2 parallel build jobs on `gpuA40x4` (each job uses its own isolated `APPTAINER_CACHEDIR=/work/nvme/bdnh/zwang83/.apptainer-cache/$SLURM_JOB_NAME` and node-local `/tmp/isb-build-$SLURM_JOB_ID`):
```bash
mkdir -p slurm/logs .apptainer/images .apptainer/volumes/isb-model-cache/hub .apptainer/volumes/isb-shared-cache/hub

sbatch --job-name=isb-build-hf slurm/delta-build.slurm \
  nnsight-hf/0-8-1 nnsight-hf/0-8-1/tp2

sbatch --job-name=isb-build-vllm slurm/delta-build.slurm \
  nnsight-vllm/0-8-1 vllm-plain/0-27-1 vllm-plain/0-28-0
```

Wait for both build jobs to complete (`squeue -u $USER`) and verify all `.sif` files exist in `.apptainer/images/`:
```bash
ls -lh .apptainer/images/isb-*.sif
```

---

## 4. Step 2 — Verify Model Cache (`isb-model-cache` & `isb-shared-cache`)

Under `ISB_LAUNCHER=apptainer`, `isb/jobs/apptainer.py` binds:
- `.apptainer/volumes/isb-model-cache:/models` by default (or `.apptainer/volumes/$ISB_MODEL_CACHE:/models` when `ISB_MODEL_CACHE` is set).
- Crucially, unlike Docker, **Apptainer automatically follows symlinks on the host filesystem as long as the symlink target is also on `/work/nvme` or `/work/hdd` and bound or accessible**, OR even simpler: bind/copy/link the `models--*` directories inside `.apptainer/volumes/isb-shared-cache/hub/` and `.apptainer/volumes/isb-model-cache/hub/`.

Verify that all 4 scaling models exist and have complete `config.json` + tokenizer + `.safetensors` files before submitting GPU jobs:
- `models--Qwen--Qwen2.5-7B-Instruct` (`14.2 GiB`)
- `models--Qwen--Qwen2.5-14B-Instruct` (`27.6 GiB`)
- `models--Qwen--Qwen2.5-72B-Instruct` (`135 GiB`)
- `models--meta-llama--Llama-3.3-70B-Instruct` (`132 GiB`, optional if also running 70B on H200)

```bash
export ISB_APPTAINER_DIR=$PWD/.apptainer
for vol in isb-model-cache isb-shared-cache; do
  for m in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-72B-Instruct; do
    dir="$ISB_APPTAINER_DIR/volumes/$vol/hub/models--${m//\//--}"
    echo "Checking $dir ..."
    ls -la "$dir/snapshots/"*/config.json "$dir/snapshots/"*/tokenizer_config.json
  done
done
```
*(If any model is missing from `.apptainer/volumes/isb-shared-cache/hub`, prefetch it on the login node with `HF_HUB_CACHE=$ISB_APPTAINER_DIR/volumes/isb-shared-cache/hub huggingface-cli download <repo>` and link/sync into `isb-model-cache/hub`.)*

---

## 5. Step 3 — Submit the Same-Family Scaling Suite to `gpuH200x8`

All Slurm stages below are already wired in `slurm/delta-compare.slurm` (including automatic `/tmp/{cache,torchinductor,triton,vllm,vllm_hook_storage}` cleanup before every `bench.py` invocation so Apptainer's `/tmp` bind never leaks caches across runs).

### 5.1 Smoke Check First (`1× H200` and `2× H200`, ~15 min)
```bash
# 2-GPU smoke check on Qwen2.5-72B (4 prompts on steering across gpu2 and tp2)
sbatch --job-name=isb-h200-smoke --partition=gpuH200x8 --exclude=gpue01 \
  --gpus=2 --cpus-per-task=32 --mem=480G --time=01:00:00 \
  slurm/delta-compare.slurm large-smoke
```
Verify `runs/nnsight-large/smoke/*/report.json` has `0` failed jobs (`python3 scripts/audit_runs.py runs/nnsight-large/smoke`).

### 5.2 Submit the Full `Qwen2.5-Instruct` (`7B → 14B → 72B`) Same-Family Scaling Jobs in Parallel
Submit both jobs simultaneously to `gpuH200x8` (`1 GPU` + `2 GPUs` = `3 H200s` total, so Slurm can co-schedule both on a single `gpuH200x8` node with isolated GPU/CPU/RAM cgroups):

```bash
# Job 1 (1x H200): Qwen2.5-7B and Qwen2.5-14B single-GPU interactive realizations (opt7b)
#                  + Qwen2.5-7B Offline, Concurrent, and Open-Loop load sweeps (tab:eval-regimes left col)
sbatch --job-name=isb-scale-qwen25-1g --partition=gpuH200x8 --exclude=gpue01 \
  --gpus=1 --cpus-per-task=16 --mem=240G --time=06:00:00 \
  slurm/delta-compare.slurm eval-scale-qwen25-1gpu

# Job 2 (2x H200): Qwen2.5-14B and Qwen2.5-72B 2-GPU interactive realizations (optlarge: gpu2 vs tp2 vs vLLM tp2/opt/taps)
#                  + Qwen2.5-72B Offline and Concurrent load sweeps (tab:eval-regimes right col)
sbatch --job-name=isb-scale-qwen25-2g --partition=gpuH200x8 --exclude=gpue01 \
  --gpus=2 --cpus-per-task=32 --mem=480G --time=08:00:00 \
  slurm/delta-compare.slurm eval-scale-qwen25-2gpu
```

### 5.3 What These Two Slurm Jobs Produce (All on Uniform `H200` Hardware!)
| Output Directory (`runs/eval-delta/...`) | Model & GPUs | Workloads & Regimes | Paper Table / Figure |
| :--- | :--- | :--- | :--- |
| `scale-opt-qwen2.5-7b` | `Qwen2.5-7B` (`1× H200`) | 5 core workloads (`interactive` + `generation`), `0-8-1`, `opt`, `opt/taps`, `vllm-plain/0-27-1` (+`/eager`) | Same-family Rung 1 (`7B @ 1 GPU`) interactive latency & `opt7b.py` report |
| `scale-opt-qwen2.5-14b` | `Qwen2.5-14B` (`1× H200`) | Same 5 core workloads (`1× H200`) | Same-family `14B @ 1 GPU` vs. `14B @ 2 GPU` crossover baseline |
| `scale-tp2-qwen2.5-14b` | `Qwen2.5-14B` (`2× H200`) | Same 5 core workloads (`gpu2`, `tp2`, `vllm/tp2`, `opt/tp2`, `opt/taps/tp2`, `plain/tp2`) | Same-family Rung 2 (`14B @ 2 GPU`) realization comparison |
| `scale-tp2-qwen2.5-72b` | `Qwen2.5-72B` (`2× H200`) | Same 5 core workloads (`gpu2`, `tp2`, `vllm/tp2`, `opt/tp2`, `opt/taps/tp2`, `plain/tp2`) | Same-family Rung 3 (`72B @ 2 GPU`) realization comparison & `optlarge.py` report |
| `scale-load-offline-7b`<br>`scale-load-concurrent-7b`<br>`scale-load-open-7b` | `Qwen2.5-7B` (`1× H200`) | `Offline` ($B \in \{1,16,64,\text{all}\}$), `Concurrent` ($C \in \{1,4,16,64\}$), `Open` ($R \in \{4,16,64\}$) | **`tab:eval-regimes` Left Column (`7B on 1× H200`)** |
| `scale-load-offline-72b`<br>`scale-load-concurrent-72b` | `Qwen2.5-72B` (`2× H200`) | `Offline` ($B \in \{1,16,64,\text{all}\}$), `Concurrent` ($C \in \{1,4,16,64\}$) | **`tab:eval-regimes` Right Column (`72B on 2× H200`)** |

---

## 6. Optional Step 4 — If Also Running the `Llama-3` (`8B → 70B → 405B`) Ladder on `8× H200`

If you also want the `Llama-3.1-8B → Llama-3.3-70B → Llama-3.1-405B` (`TP=1, 4, 8` and fixed `TP=8`) ladder on `gpuH200x8`:
1. **Wait until Steps 3.1–3.2 (`Qwen2.5` runs) finish** so those runs record the unmodified `d0ed4b9` commit hash.
2. Then add `llama3.1-8b` (`32` layers) and `llama3.1-405b` (`126` layers, `meta-llama/Llama-3.1-405B-Instruct` or `-FP8` if `810 GB` bf16 exceeds `8× 141 GB = 1,128 GB` at `0.85` utilization; note that in bf16 `405B` is `810 GB` = `101.25 GB/GPU` across `8× H200`, leaving `39.75 GB/GPU` on `141 GB` H200s!) to `isb/models.py`, and add `0-8-1/gpu8`, `0-8-1/tp8`, `0-8-1/opt/tp8`, `0-8-1/opt/taps/tp8`, `0-27-1/tp8`, `0-27-1/eager/tp8` (`gpus: 8`) to `backends/{nnsight-hf,nnsight-vllm,vllm-plain}/configs.yml`.
3. Commit that addition on a dedicated branch `feat/h200-llama-405b` and submit with `--partition=gpuH200x8 --gpus=8 --cpus-per-task=128 --mem=1920G`.

---

## 7. Step 5 — Mandatory Gate 3 Audit, HTML Reports, & Packaging Back to `hakone`

Once the Slurm jobs complete, run the forensic auditor (`scripts/audit_runs.py`) and build the HTML reports:
```bash
/work/nvme/bdnh/zwang83/envs/isb-host/bin/python scripts/audit_runs.py \
  runs/eval-delta/scale-*

# Generate the 7B, 14B, and 72B interactive realization HTML reports
/work/nvme/bdnh/zwang83/envs/isb-host/bin/python scripts/report/opt7b.py \
  runs/eval-delta/scale-opt-qwen2.5-7b runs/eval-delta/opt7b-h200.html

/work/nvme/bdnh/zwang83/envs/isb-host/bin/python scripts/report/optlarge.py \
  runs/eval-delta/scale-tp2-qwen2.5-14b runs/eval-delta/opt14b-tp2-h200.html

/work/nvme/bdnh/zwang83/envs/isb-host/bin/python scripts/report/optlarge.py \
  runs/eval-delta/scale-tp2-qwen2.5-72b runs/eval-delta/opt72b-tp2-h200.html

# Package JSON/HTML results (excluding multi-GB result.pt tensors for fast transfer, plus a full archive on /work)
tar czf eval-delta-reports.tgz --exclude='*.pt' runs/eval-delta/
```

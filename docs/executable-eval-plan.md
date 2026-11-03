# Executable Evaluation Runbook (`InterpServeBench` / `nnbench`)

> **Target Executor:** Autonomous Coding & Evaluation Agent (Claude Opus) + Reviewer/Audit Subagents
> **Primary Benchmark Checkout:** `/disk/u/zikai/interp-serve-bench/.claude/worktrees/review-17-20`
> **Commit Provenance:** Primary `eval-v1` Waves 1–3 completed on `d0ed4b9bfac87cefa2bf01c44650176c3f06a8d6`; repo updated to `c3cc4dd` (`audit_runs.py --allow-commits`). Post-audit Stage 6 tasks (`/disk/u/zikai/interp-serve-bench-paper/docs/opus-audit-handoff.md`) commit cleanly on top of `c3cc4dd` and validate with `--allow-commits`.
> **Paper Checkout:** `/disk/u/zikai/interp-serve-bench-paper`
> **Hardware & Launcher (`hakone`):** 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch), `2.0 TiB` RAM, `/disk` (`982 GB` free), `/` (`83 GB` free), native **Docker (`ISB_LAUNCHER=docker`)**.

---

## 1. Non-Negotiable Fairness & Controlled Evaluation Rules

1. **Tracked Commit Provenance (`d0ed4b9` Primary Corpus + `--allow-commits` for Stage 6 Supplements):**
   - Primary Waves 1–3 (`stage1*` through `stage5*`) completed on `d0ed4b9bfac87cefa2bf01c44650176c3f06a8d6`.
   - For **Stage 6** post-audit adapter cells, continuous subsystem sweeps, and Figure 3 CUDA-event profiling (see `/disk/u/zikai/interp-serve-bench-paper/docs/opus-audit-handoff.md`), **modifying the benchmark repo IS allowed and required**: implement the needed cells/configs/sweeps, commit cleanly before running on GPU, and pass `--allow-commits d0ed4b9,<new_commit>` to `scripts/audit_runs.py`.
   - Never silently substitute unrelated workloads or fabricate numbers when a cell fails or OOMs; report the exact hardware/software outcome.

2. **Stay on Docker (`ISB_LAUNCHER=docker`) with Image Phases:**
   - Waves 1–3 (Stages 1–5) ran in three engine waves (§3–§5).
   - On 2026-10-28 `/` (`/var/lib/docker`) has `33 GB` free with the `0.19.1` and `0.27.1` stacks and the HF images present (`53.5 GB` of images). One vLLM image stack takes `21–23 GB`.
   - Stage 6 runs in image phases: build one or a few images, run every result that needs them (functional, memory and timing), delete them, then build the next batch. Start a build only with `>= 45 GB` free and no container running; start a run only with `>= 30 GB` free. Record `df -h /` before and after every `docker rmi`.
   - **vLLM 0.19.1 and 0.27.1 phase** (images present): `vllm-hook`, `vllm-lens`, `nnsight-vllm/{default,0-8-1,...}`, `nnsight-hf/*`, `vllm-plain/{0-19-1,0-27-1}`. Then delete the `0.19.1` and `0.27.1` stacks (about `44.7 GB`).
   - **vLLM 0.20.2 and 0.28.0 phase**: build `transformer-lens-vllm` + `vllm-plain/0-20-2` and `interp-engine-vllm` + `vllm-plain/0-28-0`; run their smokes first, then every result. The HF images stay for both phases.
   - A cross-system experiment is split into one `bench.py run` per phase, each with the stock floor of its own vLLM version. `attach_plain_vllm` (`isb/jobs/score.py:41-72`) matches floors by vLLM version, execution mode and workload, so the ratios are unchanged.

3. **Phase 2 Thread Policy Preserved & 3-Tier Workload-Aware Multi-GPU Parallelization:**
   - Keep `entry.py` and `compose.yml` untouched (stock PyTorch/vLLM threads for single-process jobs; `OMP_NUM_THREADS=8` only on `ISB_RANKS > 1` HF `torchrun`).
   - `hakone` has 2 independent NUMA sockets: **Socket 0** (`GPUs 0, 1, 2, 3`, dedicated PCIe root complexes `1B, 3D, 4E, 5F`, 48 cores, 1 TiB RAM) and **Socket 1** (`GPUs 4, 5, 6, 7`, dedicated PCIe root complexes `9D, BD, CD, DD`, 48 cores, 1 TiB RAM).
   - Stage 6 concurrency (the Tier 0–2 lines below are the Waves 1–3 rules): containers are not pinned to a socket (no `cpuset` in any `compose.yml`), and within one run backends execute one after another (`isb/jobs/cli.py:175-178`), so a system's median and its floor's median are measured minutes apart. A neighbouring run that starts or stops between them changes one side of the overhead ratio.
     - **Functional runs (no reported timing):** smokes, support-table cells, correctness and equivalence, fp32 controls, TP equivalence. Up to 4 single-GPU containers (`--gpu 0`, `2`, `4`, `6`) or 2x `TP=4` (GPUs 0–3 and 4–7).
     - **Light timing** (interactive single-request steering, ablation, gen_steering, activation_patching; steering cells of the depth sweep): 2-way, one per socket (`--gpu 0` + `--gpu 4`), only after this check passes: prefill steering on `qwen2.5-7b` with `nnsight-vllm/0-8-1` and `vllm-plain/0-27-1` (eager and compiled) twice alone on GPU 0, then once while a light-timing run occupies GPU 4; accept if the paired medians and ratios lie within the spread of the two solo runs. Otherwise one at a time.
     - **Heavy timing** (logit_lens, activation_harvest, byte-volume sweep, vLLM-Hook capture cells of the depth sweep, TP 1/2/4 scaling, offline/concurrent/open regimes, J-lens optimized vs identity, Fig. 3 profiling, memory attribution with vLLM's torch profiler): one at a time, no other container on the host.
     - Image builds, `docker rmi` and `docker builder prune` run alone.
     - Each run yields only the GPUs it uses; the GPU holders (ours, removable any time) keep the others.
   - **Tier 0 (Correctness / Equivalence / Capability / Portability / FP32 Controls):** Run up to **4 GPUs in parallel** (`--gpu 0`, `--gpu 2`, `--gpu 4`, `--gpu 6`) or **2x `TP=4` in parallel** (`--gpu 0` [GPUs 0–3] and `--gpu 4` [GPUs 4–7]). Applies to `1a2-capability-waveB`, `1b3-qwen3-8b`, `1c2-fp32`, `4a-vllm-tp-equiv`, and `5c-llama70b-cov-*`.
   - **Tier 1 (Lightweight GPU-Bound Single-Request Timing — `steering`, `ablation`, `gen_steering`, `activation_patching`):** Run **2-way parallel across opposite NUMA sockets** (`--gpu 0` on Socket 0 + `--gpu 4` on Socket 1; for `TP=2`, GPUs `0–1` on Socket 0 + GPUs `4–5` on Socket 1; for `TP=4`, GPUs `0–3` on Socket 0 + GPUs `4–7` on Socket 1).
   - **Tier 2 (Heavy IPC / PCIe D2H / CPU Load Timing — `logit_lens`, `1c1-cross-system`, `3a-offline`, `3b-concurrent`, `3c-open`, `5b-llama70b-{offline,concurrent,open}`):** Run **strictly 1-at-a-time (`--gpu 0`)** with no other active containers on the host.
   - All four queue scripts (`/disk/u/zikai/.claude/jobs/d14d561b/tmp/wave1-stage1.sh`, `wave1-stage345.sh`, `wave2-all.sh`, and `wave3-all.sh`) already implement this 3-tier schedule.

4. **Strict Model-to-GPU Headroom Policy (`>= 35 GiB` Free VRAM per GPU on 80 GB A100s) & A100-First Complete Execution (No `H200` Required):**
   - **All Core Evaluation Waves (`Wave 1`, `Wave 2`, `Wave 3`) Run Completely on `hakone` (`8x A100-80GB`) Without Requiring `H200` (`Delta`):**
     - **1-GPU Runs (`--gpu 0`):** Only models `<= 14B` (`gpt2`, `smollm2-135m`, `qwen3.5-4b`, `nemotron3-nano-4b`, `qwen2.5-7b`, `qwen3-8b`, `qwen2.5-14b` — all already cached in `/disk/u/zikai/isb-model-cache/hub`). Always pass `--model qwen2.5-7b` on `nnsight-versions` and `nnsight-load-{offline,concurrent,open}` so `qwen3-32b` is never run on 1 GPU.
     - **2-GPU Runs (`*/tp2`, `*/gpu2`):** `qwen2.5-14b` and `qwen3-32b` (`~38 GiB/GPU` free headroom). Never run `qwen2.5-72b` on 2 A100s.
     - **4-GPU Runs (`*/tp4`, `*/gpu4`):** `llama3.3-70b` (`~35 GiB/GPU` free headroom across 4x A100), providing the full multi-GPU deployment & regime scaling comparison (`Stage 5A` fastest realizations, `Stage 5B` offline/concurrent/open-loop regimes, `Stage 5C` version evolution).
     - **Copy (Do Not Symlink) `Qwen3-32B` and `Llama-3.3-70B` into `/disk/u/zikai/isb-model-cache/hub`:** Because Docker containers only mount `isb-model-cache:/models`, copy `models--Qwen--Qwen3-32B` (`62 GiB`) and `models--meta-llama--Llama-3.3-70B-Instruct` (`132 GiB`) from `/disk/u/models/` into `/disk/u/zikai/isb-model-cache/hub/` using `cp -rL` before Stage 4/5 (`194 GiB` out of `982 GB` free on `/disk`).
   - **Optional Post-Wave-3 `> 100B` Scaling Extension (Evaluated After Reviewing `hakone` Results):**
     - Once Waves 1–3 complete on `hakone`, we inspect the full `7B -> 14B -> 32B -> 70B` data first. If an explicit `> 100B` checkpoint is still desired to reinforce the large-model motivation:
       - **Option A (`hakone` `8x A100-80GB = 640 GiB` HBM, zero `H200` needed):** Run a `140B–235B` `bf16` model (`Qwen3-235B-A22B` [`470 GB` -> `58.8 GB/GPU`] or `Mixtral-8x22B` [`282 GB` -> `35.3 GB/GPU`]) or `Llama-3.1-405B-FP8` (`405 GB` -> `50.6 GB/GPU`) at `TP=8` vs. `HF gpu8` directly on `hakone`.
       - **Option B (`Delta` `gpuH200x8`):** Run the standalone `eval-scale-405b` stage (`docs/delta-eval-handoff.md`) only if a dense `405B bf16` (`810 GB`) point is needed.

5. **Multi-System Tensor Parallelism (`TP > 1`) Classification Rule (Never Misclassify Exposed Shards as `UNSUPPORTED`):**
   - Official documentation and source across all 5 evaluated systems confirm that **all 5 systems support `tensor_parallel_size > 1`**, using two valid design patterns for sharded state (`ColumnParallelLinear`, `RowParallelLinear`, `ParallelLMHead`, attention heads):
     1. **Automatic In-Worker Collective Gather:** `nnsight` (`VLLMBatcher` / `VLLMFragments` auto-gathers and re-splits module `.input`/`.output`; `model.logits_processor` gathers vocab-sharded `lm_head`), `vllm-lens` (`ctx.get_parameter()` performs cross-rank collective gather; `v1.3.0` concatenates head-sharded $Q$/$K$ across TP ranks), and `interp-engine` (`vllm_capture/_tp.py` gathers sharded hook points and `decode_residuals` gathers vocab shards via `compute_logits`).
     2. **Explicit User-Invoked Gather / Per-Rank Shard Stitching:** `nnsight` HF TP (`DTensor.to_local()` returns the rank shard while `.full_tensor()` / `nnsight.modeling.tp.gather()` gathers on demand) and `vLLM-Hook` / `MIA` (each TP rank writes its head/layer shard to `hook_dir/tp_rank_<r>/`, and the user or `load_and_merge_{hs,qk}_cache` stitches the shards together).
   - Returning per-rank shards or `DTensor`s for user-side assembly is an **interface coordinate-virtualization trade-off (automatic vs. manual shard assembly)**, **never** `UNSUPPORTED`. A cell is only `SILENTLY_WRONG` under TP when an unvirtualized rank-0 slice is silently consumed as if it were the full tensor (e.g., naive single-GPU code indexing `lm_head.weight[tok]` on non-owning ranks without gathering).

6. **Ignore `nnsight 0.5.15` Everywhere:**
   - All `0-5-15` configs (`nnsight-hf/0-5-15*`, `nnsight-vllm/0-5-15*`) are excluded.

7. **Matched `sdpa`/`fp32` Reference & Version-Matched `vllm-plain` Floors (Compiled + `/eager`):**
   - Always pass `--reference nnsight-hf/sdpa` or `--reference nnsight-hf/0-8-1` (both use `bfloat16` + `sdpa`) for `bfloat16` runs (including overriding the reference on `smoke`, `nemotron`, and `fitted-jlens`), and `--reference nnsight-hf/fp32` for `fp32` controls.
   - Every timing run on a vLLM release (`0.15.1`, `0.19.1`, `0.20.2`, `0.27.1`, `0.28.0`) **must** include both `vllm-plain/<ver>` and `vllm-plain/<ver>/eager` (at the matching `tp` size) in `--backends` so `score.py` populates `overhead_vs_plain_vllm` on 100% of timed vLLM cells (enforced as a hard check by `scripts/audit_runs.py`).

---

## 2. Stage 0 Remainder — Copy `32B`/`70B` to Cache & Build Wave 1 (`0.19.1` + `0.27.1`) Images

Steps 0.1–0.4 (unit tests, `<= 14B` model downloads, `0.27.1` pilot build, and pilot run `/disk/u/zikai/isb-runs/eval-v1-pilot/20261025T194041Z-a684d934`) are **already complete**.

### Step 0.5: Copy `Qwen3-32B` and `Llama-3.3-70B-Instruct` into `/disk/u/zikai/isb-model-cache/hub`
Copy (no symlinks) from `/disk/u/models` into `/disk/u/zikai/isb-model-cache/hub` so Docker containers can load them offline in Stages 4 and 5:
```bash
for m in models--Qwen--Qwen3-32B models--meta-llama--Llama-3.3-70B-Instruct; do
  rm -f "/disk/u/zikai/isb-model-cache/hub/$m"
  if [ ! -d "/disk/u/zikai/isb-model-cache/hub/$m" ]; then
    cp -rL "/disk/u/models/$m" "/disk/u/zikai/isb-model-cache/hub/$m"
  fi
done
```

### Step 0.6: Build Wave 1 Images (`nnsight-hf` + `vLLM 0.19.1` Stack Alongside Existing `0.27.1`)
Notice we **skip** `0.20.2` (`transformer-lens-vllm`) and `0.28.0` (`interp-engine-vllm` / `nnsight-vllm/opt/taps`) during Wave 1 so only `0.27.1` (already built) and `0.19.1` (`~21 GB`) reside on `/`, keeping **`~55–60 GB` free on `/`**:
```bash
cd /disk/u/zikai/interp-serve-bench/.claude/worktrees/review-17-20
/usr/bin/sg docker -c "python3 scripts/bench.py build \
  nnsight-hf/default nnsight-hf/fp32 nnsight-hf/0-8-1/tp2 \
  nnsight-vllm/default vllm-lens/default vllm-hook/default \
  vllm-plain/0-19-1"
df -h / /disk
```

---

## 3. Engine Wave 1 — All `0.19.1` & `0.27.1` Runs (`Stage 1 Wave A`, `Stage 3`, `Stage 4`, `Stage 5A/5B`)

> **MANDATORY GATE 1 CHECK:** Spawn the **Pre-Run Setup Auditor Subagent (§7.1)** before starting Wave 1 runs.

### 3.1 Stage 1A & 1B (Wave A: `nnsight-hf`, `nnsight-vllm`, `vllm-lens`, `vllm-hook` on `0.19.1` + `0.27.1`)
```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1A-1: Smoke suite (gpt2 + smollm2-135m, all 13 workloads, all realizations, sdpa reference)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite smoke \
  --backends nnsight-hf/sdpa nnsight-vllm/default vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/sdpa --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1a-smoke"

# 1A-2 (Wave A): Full 13-workload capability sweep on Qwen2.5-7B for 0.19.1 systems (nnsight-vllm, vllm-lens, vllm-hook)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --workload logit_lens jacobian_lens steering gen_steering activation_patching gen_patching \
             ablation attention_pattern attribution_patching das jacobian_collect \
             activation_harvest gen_capture \
  --model qwen2.5-7b --realizations default \
  --backends nnsight-hf/sdpa nnsight-vllm/default vllm-lens/default vllm-hook/default \
             vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/sdpa --gpu 0 --timeout 3600 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1a-capability-waveA"

# 1B-1: Nemotron-3-Nano-4B hybrid Mamba-2 + attention suite (with sdpa reference + plain floors)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite nemotron \
  --backends nnsight-hf/sdpa nnsight-vllm/default vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/sdpa --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1b-nemotron"

# 1B-2: Qwen3.5-4B fitted Jacobian lens suite (with sdpa reference + plain floors)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite fitted-jlens \
  --backends nnsight-hf/sdpa nnsight-vllm/default vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/sdpa --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1b-jlens"

# 1B-3 (Wave A): Qwen3-8B portability for 0.19.1 systems
for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
  /usr/bin/sg docker -c "python3 scripts/bench.py run \
    --workload ${spec%%|*} --model qwen3-8b --data ${spec##*|} \
    --regimes interactive generation --realizations default \
    --backends nnsight-hf/sdpa nnsight-vllm/default vllm-lens/default vllm-hook/default \
               vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
    --reference nnsight-hf/sdpa --gpu 0 --timeout 3600 \
    --out /disk/u/zikai/isb-runs/eval-v1/stage1b-qwen3-8b-waveA"
done
```

### 3.2 Stage 1C (Wave A: Cross-System Latency, FP32 Control, & `0.19.1`/`0.27.1` Realization Headroom)
```bash
# 1C-1 (Wave A): Cross-System Suite on Qwen2.5-7B for 0.19.1 systems
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite cross-system \
  --backends nnsight-hf/sdpa nnsight-vllm/default vllm-lens/default vllm-hook/default \
             vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
  --reference nnsight-hf/sdpa --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1c-cross-system-waveA"

# 1C-2 (Wave A): FP32 Precision Control on Qwen2.5-7B for nnsight-hf/fp32 and nnsight-vllm/fp32
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite cross-system \
  --backends nnsight-hf/fp32 nnsight-vllm/fp32 \
  --reference nnsight-hf/fp32 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1c-fp32-waveA"

# 1C-3 (Wave A): Realization Headroom on Qwen2.5-7B for 0.27.1 (nnsight 0-8-1 / opt / opt/taps) and 0.19.1 (vllm-lens/opt, vllm-hook/disk)
for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
  /usr/bin/sg docker -c "python3 scripts/bench.py run \
    --workload ${spec%%|*} --model qwen2.5-7b --data ${spec##*|} \
    --regimes interactive generation --realizations default \
    --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 nnsight-vllm/0-8-1/opt \
               nnsight-vllm/0-8-1/opt/taps vllm-lens/opt vllm-hook/disk \
               vllm-plain/0-19-1 vllm-plain/0-19-1/eager \
               vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
    --reference nnsight-hf/0-8-1 --gpu 0 --timeout 3600 \
    --out /disk/u/zikai/isb-runs/eval-v1/stage1c-opt7b-waveA"
done
```

> **MANDATORY GATE 3 CHECK:** Run `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage1*` and verify `0` errors before continuing.

### 3.3 Stage 3: Serving Load Regimes (`Offline`, `Concurrent`, `Open-Loop`) on `Qwen2.5-7B` (`0.27.1` Stack — Already Built!)
```bash
# 3A: Offline Dataset Sweep (256 units, B in {1, 16, 64, all}) on Qwen2.5-7B
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite nnsight-load-offline --model qwen2.5-7b \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 nnsight-vllm/0-8-1/opt \
             vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage3a-offline"

# 3B: Closed-Loop Concurrent Serving (128 units, C in {1, 4, 16, 64}) on Qwen2.5-7B
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite nnsight-load-concurrent --model qwen2.5-7b \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 nnsight-vllm/0-8-1/async \
             vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage3b-concurrent"

# 3C: Open-Loop Poisson Arrivals (128 units, R in {4, 16, 64} req/s) on Qwen2.5-7B
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite nnsight-load-open --model qwen2.5-7b \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1/async \
             vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage3c-open"
```

> **MANDATORY GATE 3 CHECK:** Run `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage3*`.

### 3.4 Stage 4: Multi-GPU Parallelism (`TP=1,2,4` & `TP=2` vs. `gpu2`) on `Qwen2.5-14B` & `Qwen3-32B` (`0.27.1` Stack — Already Built!)
```bash
# 4A: Pure vLLM Tensor-Parallel Equivalence on Qwen2.5-14B (TP=1 vs. TP=2 vs. TP=4; no vllm-plain so each TP size is compared purely for equivalence against 1-GPU vLLM)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite parallel \
  --backends nnsight-vllm/0-8-1 nnsight-vllm/0-8-1/tp2 nnsight-vllm/0-8-1/tp4 \
             nnsight-vllm/0-8-1/opt/tp2 nnsight-vllm/0-8-1/opt/taps/tp2 \
  --reference nnsight-vllm/0-8-1 --comparison equivalence --gpu 0 --timeout 3600 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage4a-vllm-tp-equiv"

# 4B: 2-GPU Realization Comparison (HF gpu2 vs. HF tp2 vs. vLLM tp2/opt/taps + vllm-plain 0-27-1/tp2 floors) on Qwen2.5-14B and Qwen3-32B
for model in qwen2.5-14b qwen3-32b; do
  for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
    /usr/bin/sg docker -c "python3 scripts/bench.py run \
      --workload ${spec%%|*} --model $model --data ${spec##*|} \
      --regimes interactive generation --realizations default \
      --backends nnsight-hf/0-8-1/gpu2 nnsight-hf/0-8-1/tp2 \
                 nnsight-vllm/0-8-1/tp2 nnsight-vllm/0-8-1/opt/tp2 nnsight-vllm/0-8-1/opt/taps/tp2 \
                 vllm-plain/0-27-1/tp2 vllm-plain/0-27-1/eager/tp2 \
      --reference nnsight-hf/0-8-1/gpu2 --gpu 0 --timeout 3600 \
      --out /disk/u/zikai/isb-runs/eval-v1/stage4b-tp2-$model"
  done
done
```

> **MANDATORY GATE 3 CHECK:** Run `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage4*`.

### 3.5 Stage 5A & 5B: `Llama-3.3-70B-Instruct` on `4x A100` (`0.27.1` Stack — Already Built!)
Runs the `0-8-1` (`vLLM 0.27.1`) 70B experiments with matching `vllm-plain/0-27-1/tp4` and `vllm-plain/0-27-1/eager/tp4` floors:
```bash
# 5A: Llama-3.3-70B-Instruct (4x A100) — Fastest Realizations
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite llama70b-fastest \
  --backends nnsight-hf/0-8-1/gpu4 nnsight-hf/0-8-1/tp4 \
             nnsight-vllm/0-8-1/tp4 nnsight-vllm/0-8-1/opt/tp4 \
             nnsight-vllm/0-8-1/opt/taps/tp4 \
             vllm-plain/0-27-1/tp4 vllm-plain/0-27-1/eager/tp4 \
  --reference nnsight-hf/0-8-1/gpu4 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage5a-llama70b-fastest"

# 5B-1: Llama-3.3-70B-Instruct — Offline Load Pattern (with 0-27-1/tp4 plain floors)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite llama70b-offline \
  --backends nnsight-hf/0-8-1/gpu4 nnsight-vllm/0-8-1/tp4 \
             vllm-plain/0-27-1/tp4 vllm-plain/0-27-1/eager/tp4 \
  --reference nnsight-hf/0-8-1/gpu4 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage5b-llama70b-offline"

# 5B-2: Llama-3.3-70B-Instruct — Concurrent Load Pattern (with 0-27-1/tp4 plain floors)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite llama70b-concurrent \
  --backends nnsight-hf/0-8-1/gpu4 nnsight-vllm/0-8-1/async/tp4 \
             vllm-plain/0-27-1/tp4 vllm-plain/0-27-1/eager/tp4 \
  --reference nnsight-hf/0-8-1/gpu4 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage5b-llama70b-concurrent"

# 5B-3: Llama-3.3-70B-Instruct — Open-Loop Load Pattern (with 0-27-1/tp4 plain floors)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite llama70b-open \
  --backends nnsight-hf/0-8-1/gpu4 nnsight-vllm/0-8-1/async/tp4 \
             vllm-plain/0-27-1/tp4 vllm-plain/0-27-1/eager/tp4 \
  --reference nnsight-hf/0-8-1/gpu4 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage5b-llama70b-open"
```

> **MANDATORY GATE 3 CHECK:** Run `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage5*`.

---

## 4. Engine Wave 2 — Swap `0.19.1` -> `0.20.2` (`transformer-lens-vllm`) & `0.28.0` (`interp-engine-vllm`) for Stage 1 Wave B

Once Engine Wave 1 passes Gate 3, remove the `0.19.1` images (`+21 GB` reclaimed) and build `0.20.2` (`transformer-lens-vllm`) + `0.28.0` (`interp-engine-vllm`):

```bash
# Prune 0.19.1 images (keeping nnsight-hf/* and 0-8-1/0-27-1 intact)
/usr/bin/sg docker -c "docker rmi \
  isb-nnsight-vllm-default:local isb-vllm-lens-default:local \
  isb-vllm-hook-default:local isb-vllm-plain-0-19-1:local \
  vllm/vllm-openai:v0.19.1 2>/dev/null || true"
/usr/bin/sg docker -c "docker builder prune -af"
df -h /

# Build 0.20.2 (transformer-lens-vllm) and 0.28.0 (interp-engine-vllm)
/usr/bin/sg docker -c "python3 scripts/bench.py build \
  transformer-lens-vllm/default interp-engine-vllm/default \
  vllm-plain/0-20-2 vllm-plain/0-28-0"
df -h /
```

*(Note: If `/` free space is `< 35 GB` after building `0.20.2`, run `transformer-lens-vllm` first, `docker rmi` `0.20.2`, and then build/run `0.28.0` `interp-engine-vllm` so `/` always stays above `55 GB` free!)*

```bash
# 1A-2 (Wave B): Full 13-workload capability sweep on Qwen2.5-7B for transformer-lens-vllm (0.20.2) and interp-engine-vllm (0.28.0)
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --workload logit_lens jacobian_lens steering gen_steering activation_patching gen_patching \
             ablation attention_pattern attribution_patching das jacobian_collect \
             activation_harvest gen_capture \
  --model qwen2.5-7b --realizations default \
  --backends nnsight-hf/sdpa transformer-lens-vllm/default interp-engine-vllm/default \
             vllm-plain/0-20-2 vllm-plain/0-20-2/eager \
             vllm-plain/0-28-0 vllm-plain/0-28-0/eager \
  --reference nnsight-hf/sdpa --gpu 0 --timeout 3600 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1a-capability-waveB"

# 1B-3 (Wave B): Qwen3-8B portability for transformer-lens-vllm and interp-engine-vllm
for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
  /usr/bin/sg docker -c "python3 scripts/bench.py run \
    --workload ${spec%%|*} --model qwen3-8b --data ${spec##*|} \
    --regimes interactive generation --realizations default \
    --backends nnsight-hf/sdpa transformer-lens-vllm/default interp-engine-vllm/default \
               vllm-plain/0-20-2 vllm-plain/0-20-2/eager \
               vllm-plain/0-28-0 vllm-plain/0-28-0/eager \
    --reference nnsight-hf/sdpa --gpu 0 --timeout 3600 \
    --out /disk/u/zikai/isb-runs/eval-v1/stage1b-qwen3-8b-waveB"
done

# 1C-1 (Wave B): Cross-System Suite on Qwen2.5-7B for transformer-lens-vllm and interp-engine-vllm
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite cross-system \
  --backends nnsight-hf/sdpa transformer-lens-vllm/default interp-engine-vllm/default \
             vllm-plain/0-20-2 vllm-plain/0-20-2/eager \
             vllm-plain/0-28-0 vllm-plain/0-28-0/eager \
  --reference nnsight-hf/sdpa --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1c-cross-system-waveB"

# 1C-2 (Wave B): FP32 Precision Control on Qwen2.5-7B for transformer-lens-vllm/fp32
/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite cross-system \
  --backends nnsight-hf/fp32 transformer-lens-vllm/fp32 \
  --reference nnsight-hf/fp32 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage1c-fp32-waveB"

# 1C-3 (Wave B): Realization Headroom on Qwen2.5-7B for transformer-lens-vllm/opt (0.20.2) and interp-engine-vllm/static (0.28.0)
for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
  /usr/bin/sg docker -c "python3 scripts/bench.py run \
    --workload ${spec%%|*} --model qwen2.5-7b --data ${spec##*|} \
    --regimes interactive generation --realizations default \
    --backends nnsight-hf/0-8-1 transformer-lens-vllm/opt interp-engine-vllm/static \
               vllm-plain/0-20-2 vllm-plain/0-20-2/eager \
               vllm-plain/0-28-0 vllm-plain/0-28-0/eager \
    --reference nnsight-hf/0-8-1 --gpu 0 --timeout 3600 \
    --out /disk/u/zikai/isb-runs/eval-v1/stage1c-opt7b-waveB"
done
```

> **MANDATORY GATE 3 CHECK:** Run `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage1*`, then prune `0.20.2` and `0.28.0` images:
```bash
/usr/bin/sg docker -c "docker rmi \
  isb-transformer-lens-vllm-default:local isb-interp-engine-vllm-default:local \
  isb-vllm-plain-0-20-2:local isb-vllm-plain-0-28-0:local \
  vllm/vllm-openai:v0.20.2 vllm/vllm-openai:v0.28.0 2>/dev/null || true"
/usr/bin/sg docker -c "docker builder prune -af"
```

---

## 5. Engine Wave 3 — Legacy Release Evolution (`0.4.6` -> `0.7` -> `0.8.1`) (`Stage 2A`)

With `0.20.2` and `0.28.0` pruned (`~83 GB` free on `/`), build `0-7` (`vLLM 0.15.1`) and `0-4-6`, run `Stage 2A` (`nnsight-versions` on `Qwen2.5-7B`), audit, and prune `0-4-6`/`0-15-1`:

```bash
/usr/bin/sg docker -c "python3 scripts/bench.py build \
  nnsight-hf/0-7 nnsight-hf/0-4-6 nnsight-hf/0-4-6/era \
  nnsight-vllm/0-7 vllm-plain/0-15-1"

/usr/bin/sg docker -c "python3 scripts/bench.py run \
  --suite nnsight-versions --model qwen2.5-7b \
  --backends nnsight-hf/0-8-1 nnsight-vllm/0-8-1 \
             nnsight-hf/0-7 nnsight-vllm/0-7 \
             nnsight-hf/0-4-6 nnsight-hf/0-4-6/era \
             vllm-plain/0-15-1 vllm-plain/0-15-1/eager \
             vllm-plain/0-27-1 vllm-plain/0-27-1/eager \
  --reference nnsight-hf/0-8-1 --gpu 0 \
  --out /disk/u/zikai/isb-runs/eval-v1/stage2a-nnsight-versions"

python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1/stage2a-nnsight-versions

/usr/bin/sg docker -c "docker rmi \
  isb-nnsight-hf-0-4-6:local isb-nnsight-hf-0-4-6-era:local \
  isb-nnsight-hf-0-7:local isb-nnsight-vllm-0-7:local \
  isb-vllm-plain-0-15-1:local vllm/vllm-openai:v0.15.1 2>/dev/null || true"
/usr/bin/sg docker -c "docker builder prune -af"
```

---

## 6. Stage 6 — Post-Audit Targeted Reruns, 5-System Figure 2 Completion, Continuous Subsystem Sweeps & Direct 4-Stage Profiling

Waves 1–3 (§3–§5 above) are **completed**. Execute **Stage 6** following the audited instructions in [`/disk/u/zikai/interp-serve-bench-paper/docs/opus-audit-handoff.md`](file:///disk/u/zikai/interp-serve-bench-paper/docs/opus-audit-handoff.md) and [`/disk/u/zikai/interp-serve-bench-paper/docs/fig3-profiling-plan.md`](file:///disk/u/zikai/interp-serve-bench-paper/docs/fig3-profiling-plan.md):

0. **Phase 1.0 (HIGHEST PRIORITY) — Complete All 5 Engines for Figure 2 (`eval_fig_1_{op1,op2a,op2b}.pdf` — Offline, Online & Model Scale; see `Task 1.0` in `opus-audit-handoff.md`):**
   - In Waves 1–3, `stage3a-offline`, `stage3b-concurrent`, `stage3c-open`, `stage4b-tp2-*`, and `stage5a-llama70b-fastest` only ran `nnsight-hf` and `nnsight-vllm`.
   - Run the remaining four systems (`vllm-lens/{default,opt}`, `vllm-hook/{default,disk}`, `interp-engine-vllm/{default,static}`, `transformer-lens-vllm/{default,opt,batched}`) across all three rows of Figure 2 inside the two Docker image waves (`0.19.1` first, then `0.20.2` + `0.28.0`):
     - **Row 1 (Offline $B \in \{1, 16, 64, 256\}$ on `qwen2.5-7b`):** `--workload logit_lens steering ablation gen_steering --model qwen2.5-7b --regimes offline --offline 256:1,16,64,all --out /disk/u/zikai/isb-runs/eval-v1/stage3a-offline`.
     - **Row 2 (Online `open` $R \in \{4, 16, 64\}$ and `concurrent` $C \in \{1, 4, 16, 64\}$ on `qwen2.5-7b`):** `--suite nnsight-load-open` (`--out .../stage3c-open`) and `--suite nnsight-load-concurrent` (`--out .../stage3b-concurrent`). (`InterpEngineBackend` in `backends/interp-engine-vllm/backend.py` declares `concurrent_requests = True`.)
     - **Row 3 (Multi-GPU Model Scale `14B` `tp2`, `32B` `tp2`, `70B` `tp4`):** `vllm-lens/{tp2,opt/tp2}`, `vllm-hook/{tp2,disk/tp2}`, `interp-engine-vllm/{tp2,static/tp2}`, `transformer-lens-vllm/{tp2,opt/tp2}` on `qwen2.5-14b` and `qwen3-32b` (`--out .../stage4b-tp2-$model`), and the corresponding `tp4` configs on `--suite llama70b-fastest` (`--out .../stage5a-llama70b-fastest`), with version-matched `vllm-plain` compiled and `/eager` floors.
1. **Phase 1.1 — Missing Support-Table Adapter Cells, Ablation Scorer Fix & 7B `jacobian_collect` `dim_batch` Fix:**
   - Wire `jacobian_lens` (`llama`) in `backends/{vllm-lens,interp-engine-vllm,transformer-lens-vllm,vllm-hook}/cells.py` (and `cells_opt.py`).
   - Wire `activation_harvest` (`llama`) in `backends/vllm-hook/cells.py` (`_hs_mode = "all_tokens"`, 1-based `_output_layers = [layer + 1]`, restoring `_hs_mode = "last_token"` in `finally:`).
   - Wire `gen_patching` (`llama`, `new_tokens=5`, step-0 prefill transplant) in `backends/vllm-lens/cells.py` (and `cells_opt.py`).
   - Wire `attention_pattern` (`llama`) in `isb/methodologies/attention_pattern.py` (using `attn_implementation: eager` on `nnsight-hf`, and separating direct FlashAttention weight readout `Err`/`Unsupp` from pre-kernel Q/K capture + off-kernel $QK^\top$ reconstruction with GQA `repeat_kv`, slicing fused `qkv_proj` on `nnsight-vllm`).
   - Note: Interactive `ablation` (`INVALID_REFERENCE` from the automated 0.2 TV threshold) was manually verified from stored `result.pt` tensors (`TV = 0.012–0.028`, `top-1 = 0.875–1.0` vs. `0.087–0.18` unablated), so no ablation reruns are required. Workload memory on vLLM-backed systems is measured inside the engine workers with vLLM's own tools: every timed cell resets and reads `torch.cuda.max_memory_allocated()` in each worker through `LLM.collective_rpc` (outside the timed region; a pickled function needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, `vllm/v1/serial_utils.py`) and reports it per rank above the same system's unperturbed request. Per-op attribution comes from vLLM's torch profiler with `ProfilerConfig.torch_profiler_with_memory=True` (`LLM.start_profile()`/`stop_profile()`) in a heavy-timing slot. Startup memory comes from vLLM's `Model loading took` and `Available KV cache memory` log lines. No matched 2048-token stock configs.
   - Lower `dim_batch` (`96 -> 16`) on `jacobian_collect_hf` (`isb/methodologies/jacobian_collect.py`) so 7B batched VJP fits on `1x A100-80GB`.
   - Profile and add an optimized realization for `Qwen3.5-4B` fitted `jacobian_lens` on `nnsight-vllm` (`1,261.88 ms` vs. `110.56 ms` identity).
2. **Phase 1.2 — TransformerLens Batched Evaluation (`enable_batching=True`):**
   - Add a `batched` config (`enable_batching: true, enable_position_interventions: false`, which forces `enforce_eager=True` and disallows `TP > 1` / `PP > 1`) in `backends/transformer-lens-vllm/configs.yml:8` alongside compiled single-prompt `default`/`opt`.
3. **Phase 1.3 — Continuous Subsystem Sweeps (Fig. 4a–c) & Fig. 2d Supplement:**
   - Fix `spec_taps(spec)` in `backends/nnsight-vllm/backend_opt.py:81-82` to support `p.get("layers")`, and add a multi-layer `layers` parameter to the steering cells of the included editing backends (`nnsight-vllm`, `transformer-lens-vllm`, `vllm-lens`; `interp-engine-vllm` is excluded from the layer sweep per user decision).
   - Update `backends/{transformer-lens-vllm,interp-engine-vllm}/cells.py` to return harvested tensors in `activation_harvest` and wire `returned_bytes` into `isb/runner.py`.
   - Run the containerized `bench.py` hooked-layer depth sweep ($L \in \{1, 4, 7, 14, 21, 28\}$ prefill steering + `vllm-hook` capture, excluding `interp-engine-vllm`) for Fig. 4a, byte-volume sweep (`0 MB` to `220 MB` `bf16` / `440 MB` `fp32` at $L=28, B=16$ on `WikiText`, reporting measured `returned_bytes` in the paper) for Fig. 4b, cross-system `TP in {1, 2, 4}` scaling on `Qwen2.5-14B` for Fig. 4c (plus profiling `nnsight-vllm/0-8-1/opt/taps` TP=4 logit lens), and `gen_steering` offline $B \in \{64, 256\}$ for Fig. 2d.
4. **Phase 2 — Direct 4-Stage Span Profiling for Figure 3 (`eval_fig_2.pdf`):**
   - Execute `/disk/u/zikai/interp-serve-bench-paper/docs/fig3-profiling-plan.md` (`isb/perf/phase_profiler.py` using deferred `torch.cuda.Event(enable_timing=True)` + `perf_counter_ns`, writing records to `/disk/u/zikai/isb-runs/eval-v1/latency-breakdown-profiling/`).

---

## 7. Multi-Agent Reviewer & Audit Architecture (Before / During / After Runs)

### 7.1 Gate 1 — Pre-Run Setup Auditor Prompt (Spawn Before Each Wave)
> **Subagent Role:** `Pre-Run Setup Auditor`
> **Prompt:**
> "Audit `/disk/u/zikai/interp-serve-bench/.claude/worktrees/review-17-20` and host state before Wave `<N>` launches. Verify and report PASS/FAIL on all 6 items:
> 1. `git status --porcelain` has no uncommitted tracked modifications under `isb/` or `backends/` (all Stage 6 changes are cleanly committed before launching runs).
> 2. `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv` shows `< 100 MiB` and `0%` utilization across all 8 GPUs.
> 3. `docker ps` shows no active `isb-*` or `dev-*` containers.
> 4. `df -h / /disk` shows at least `30 GB` free on `/` and `200 GB` free on `/disk`.
> 5. Every model targeted by the wave resolves a real `config.json` file (not a broken symlink) inside `/disk/u/zikai/isb-model-cache/hub` and leaves `>= 35 GiB` of free VRAM per GPU (`<= 14B` on 1 GPU, `<= 32B` on 2 GPUs, `70B` on 4 GPUs).
> 6. Every planned `bench.py run` command uses an `sdpa` or `fp32` HF `--reference` (or vLLM for pure TP equivalence), includes both compiled and `/eager` `vllm-plain` floors for timing runs, and excludes `0-5-15`.
> Return `GATE 1 PASS` or `GATE 1 BLOCK: <reason>`."

### 7.2 Gate 2 — Live Watchdog Auditor Prompt (Spawn 2–3 Minutes Into Long Suites)
> **Subagent Role:** `Live Telemetry Watchdog`
> **Prompt:**
> "Inspect the currently running `isb-*` container and host telemetry without disturbing the run:
> 1. `nvidia-smi`: Confirm only the expected GPU index(es) have active processes and no foreign PID is sharing the machine.
> 2. Active `execution.log`: Locate the newest `execution.log` under `/disk/u/zikai/isb-runs/eval-v1/` and check the last 30 lines for `Error in memory profiling`, `Free memory on device ... less than desired`, `CUDA out of memory`, or a hung process.
> 3. Disk space: Confirm `df -h / /disk` has `> 25 GB` free on `/`.
> Return `GATE 2 HEALTHY` or `GATE 2 ALERT: <finding>`."

### 7.3 Gate 3 — Post-Stage Forensic Auditor (`scripts/audit_runs.py`)
After every stage completes, run `python3 scripts/audit_runs.py --allow-commits d0ed4b9,$(git rev-parse --short HEAD) /disk/u/zikai/isb-runs/eval-v1/<stage>` and spawn a read-only **Post-Stage Forensic Reviewer Subagent** to verify:
1. `plan.status == "completed"` and every `execution.json` has `status == "completed"` and `exit_code == 0`.
2. Global `source.commit` is in the allowed commit set (`d0ed4b9` for Waves 1–3; the committed Stage 6 SHA for Stage 6 runs) and `git_dirty == False`.
3. `gpu_release.baseline_mib < 100` on 100% of jobs.
4. `provenance.model_identity` (`repo`, `revision`, `vocab_sha256`) matches across all backends within each experiment.
5. Zero `JOB_FAILED`, `NO_REFERENCE`, `INCOMPATIBLE`, or `INVALID_OUTPUT` states (`INVALID_REFERENCE` on weak-effect layer-1 `ablation` rows is logged as an expected warning).
6. Every timed vLLM cell has `overhead_vs_vanilla` and (whenever `vllm-plain` is in `plan.backends`) `overhead_vs_plain_vllm` populated (`audit_runs.py` enforces this as a hard error).

---

## 7. Checkable Deliverables

1. **Complete Raw Run Hierarchy (`/disk/u/zikai/isb-runs/eval-v1/stage*`):**
   - `plan.json`, `report.json`, and `experiments/<exp_id>/<system>/<config>/{execution.json,result.json,execution.log,cleanup.log}` across all waves.
2. **Zero-Error Gate 3 Audit Pass:**
   - `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1-pilot /disk/u/zikai/isb-runs/eval-v1` exits with **code `0`** (`GATE 3 AUDIT PASSED: ... with 0 errors`).
3. **Self-Contained Interactive HTML Reports (`/disk/u/zikai/isb-runs/eval-v1/deliverables/`):**
   ```bash
   mkdir -p /disk/u/zikai/isb-runs/eval-v1/deliverables
   python3 scripts/manager.py --dir /disk/u/zikai/isb-runs/eval-v1 \
     --export /disk/u/zikai/isb-runs/eval-v1/deliverables/eval-v1-site.html
   python3 scripts/report/opt7b.py /disk/u/zikai/isb-runs/eval-v1/stage1c-opt7b-waveA \
     /disk/u/zikai/isb-runs/eval-v1/deliverables/opt7b-report.html
   python3 scripts/report/optlarge.py /disk/u/zikai/isb-runs/eval-v1/stage4b-tp2-qwen3-32b \
     /disk/u/zikai/isb-runs/eval-v1/deliverables/opt32b-tp2-report.html
   ```
4. **1-to-1 Paper Table & Figure Traceability (`/disk/u/zikai/interp-serve-bench-paper/`):**
   - Every cell in `tab:eval-support`, `tab:eval-latency`, `tab:eval-regimes`, `tab:eval-parallel`, and `tab:eval-micro` in [`sections/05-evaluation.tex`](file:///disk/u/zikai/interp-serve-bench-paper/sections/05-evaluation.tex) traces directly to the verified `report.json` files.
5. **Reproducibility Metadata Tarball (`eval-v1-metadata.tgz`):**
   ```bash
   tar czf /disk/u/zikai/isb-runs/eval-v1/deliverables/eval-v1-metadata.tgz \
     --exclude="*.pt" /disk/u/zikai/isb-runs/eval-v1
   ```

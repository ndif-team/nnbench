#!/bin/bash
# nnbench PP/TP equivalence smoke (single node): vLLM (1,1) control vs vLLM (2,1) candidate,
# scored by nnbench's own oracle (GT2 = same intervention at tp=1,pp=1). No HF reference.
# Validates the in-process sequential-engine path (control engine, then candidate engine).
#SBATCH --job-name=nnbench-pp-smoke
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pp_smoke_%j.out
#SBATCH --error=slurm/logs/pp_smoke_%j.err

set -euo pipefail
source /work/nvme/bdnh/zwang83/etc/profile.d/conda.sh
conda activate bench-pp
export HF_HOME=/work/nvme/bdnh/zwang83/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1

cd /work/nvme/bdnh/zwang83/nnbench
python scripts/bench.py --spec logit_lens_gpt2 --pp 2

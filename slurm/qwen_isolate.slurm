#!/bin/bash
# Isolate the two failure mechanisms on the SAME untied model (Qwen2.5-14B, tie_word_embeddings=false),
# logit_lens_qwen, candidate scored vs single-GPU (1,1):
#   PP isolation  -> --pp 2 --tp 1 : lm_head only on last stage -> PPMissingLayer -> expect CRASH
#   TP isolation  -> --pp 1 --tp 2 : lm_head vocab-sharded, present -> expect RUN (partial/■wrong), no crash
#SBATCH --job-name=qwen-isolate
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/qwen_isolate_%j.out
#SBATCH --error=slurm/logs/qwen_isolate_%j.err

set -uo pipefail
source /work/nvme/bdnh/zwang83/etc/profile.d/conda.sh
conda activate bench-pp
export HF_HOME=/work/nvme/bdnh/zwang83/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1

cd /work/nvme/bdnh/zwang83/nnbench
reap(){ pkill -9 -u "$USER" -f "vllm.v1.engine.core" 2>/dev/null||true; \
        pkill -9 -u "$USER" -f "nnsight.modeling.vllm.workers" 2>/dev/null||true; \
        pkill -9 -u "$USER" -f "VLLM::" 2>/dev/null||true; sleep 10; }

echo "==================== ISOLATION: PP only (pp=2, tp=1) ===================="
python scripts/bench.py --spec logit_lens_qwen --pp 2 --tp 1 --gpu-mem 0.90 --max-model-len 2048 \
  || echo "!! pp2tp1 exited non-zero"
reap

echo "==================== ISOLATION: TP only (pp=1, tp=2) ===================="
python scripts/bench.py --spec logit_lens_qwen --pp 1 --tp 2 --gpu-mem 0.90 --max-model-len 2048 \
  || echo "!! pp1tp2 exited non-zero"
reap
echo "==================== ISOLATION DONE ===================="

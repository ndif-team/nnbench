#!/bin/bash
# Large-model TP/PP equivalence: Qwen2.5-14B-Instruct (bf16), candidate tp=2 pp=2 (4 GPUs, single
# node, mp executor) scored against the single-GPU (1,1) control (GT2). Sweeps the ported specs.
# attention_pattern/attribution_patching are intentionally omitted (no llama cell; vLLM has no
# attn-read/autograd) — the "naturally not supported" set.
#SBATCH --job-name=qwen-tp2pp2
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/qwen_tp2pp2_%j.out
#SBATCH --error=slurm/logs/qwen_tp2pp2_%j.err

set -uo pipefail
source /work/nvme/bdnh/zwang83/etc/profile.d/conda.sh
conda activate bench-pp
export HF_HOME=/work/nvme/bdnh/zwang83/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3   # control (1,1) uses one; candidate (tp2pp2) uses all four

cd /work/nvme/bdnh/zwang83/nnbench
SPECS="${PP_SPECS:-logit_lens_qwen steering_qwen ablation_qwen activation_patching_qwen gen_steering_qwen}"
for spec in $SPECS; do
  echo "==================== SPEC: $spec ===================="
  python scripts/bench.py --spec "$spec" --pp 2 --tp 2 --gpu-mem 0.90 --max-model-len 2048 \
    || echo "!! spec $spec exited non-zero"
  # A deadlocked candidate engine (EngineDeadError from the TP/PP manual-unembed block) leaves zombie
  # vLLM workers holding GPU memory, which makes the NEXT spec OOM at engine init. Reap them so each
  # spec starts on a clean GPU. Safe here: no vLLM should be live between specs (bench.py has exited).
  pkill -9 -u "$USER" -f "vllm.v1.engine.core" 2>/dev/null || true
  pkill -9 -u "$USER" -f "nnsight.modeling.vllm.workers" 2>/dev/null || true
  pkill -9 -u "$USER" -f "VLLM::" 2>/dev/null || true
  pkill -9 -u "$USER" -f "from multiprocessing.spawn" 2>/dev/null || true
  sleep 10
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; echo
done
echo "==================== QWEN SWEEP DONE ===================="

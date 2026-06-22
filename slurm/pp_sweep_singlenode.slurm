#!/bin/bash
# nnbench PP=2 single-node sweep across all gpt2 specs: each scored vLLM (2,1) vs (1,1) (GT2).
# One fresh bench.py process per spec (clean vLLM state between specs).
#SBATCH --job-name=nnbench-pp-sweep
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/pp_sweep_%j.out
#SBATCH --error=slurm/logs/pp_sweep_%j.err

set -uo pipefail
source /work/nvme/bdnh/zwang83/etc/profile.d/conda.sh
conda activate bench-pp
export HF_HOME=/work/nvme/bdnh/zwang83/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1

cd /work/nvme/bdnh/zwang83/nnbench
SPECS="${PP_SPECS:-steering_gpt2 ablation_gpt2 activation_patching_gpt2 attention_pattern_gpt2 attribution_patching_gpt2 gen_steering_gpt2}"
for spec in $SPECS; do
  echo "==================== SPEC: $spec ===================="
  python scripts/bench.py --spec "$spec" --pp 2 || echo "!! spec $spec exited non-zero"
done
echo "==================== SWEEP DONE ===================="

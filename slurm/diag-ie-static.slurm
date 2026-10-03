#!/bin/bash
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --exclude=gpue01,gpub003,gpub010,gpub025,gpub037,gpub050,gpub069,gpub088
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
cd /work/nvme/bdnh/zwang83/nnbench
A=$PWD/.apptainer
for b in vllm vllm-static; do
  apptainer exec --nv --cleanenv --no-home --bind $PWD --bind /tmp --bind $A/volumes/isb-model-cache:/models --pwd $PWD \
    --env PYTHONPATH=$PWD,USER=isb,HF_HOME=/models,HF_HUB_CACHE=/models/hub,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
    --env XDG_CACHE_HOME=/tmp/cache,VLLM_CACHE_ROOT=/tmp/vllm,TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor,TRITON_CACHE_DIR=/tmp/triton,FLASHINFER_WORKSPACE_BASE=/tmp,CUDA_VISIBLE_DEVICES=0 \
    $A/images/isb-interp-engine_local.sif python3 slurm/diag_ie_static.py runs/ie-static-diag/$b.pt $b 2>&1 | grep -v "^INFO\|^(EngineCore" | tail -30
  sleep 20
done

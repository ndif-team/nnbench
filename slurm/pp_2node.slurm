#!/bin/bash
# nnbench PP across TWO nodes via vLLM's stock Ray executor.
#   candidate = vLLM PP=2 (one stage per node, forced cross-node: 1 GPU/node in Ray)
#   control   = vLLM (1,1) single-GPU, run in-process on a separate head GPU
# nnbench's oracle scores candidate vs control (GT2 = same intervention at tp=1,pp=1). No HF.
#
# Submit: sbatch slurm/pp_2node.slurm
#SBATCH --job-name=nnbench-pp-2node
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --output=slurm/logs/pp_2node_%j.out
#SBATCH --error=slurm/logs/pp_2node_%j.err

# NOTE: no `-e` — this is an orchestration script full of intentional best-effort steps
# (backgrounded `srun &`, `ray status` polling whose `grep` returns non-zero until the cluster is
# ready, `ray stop || true`). `set -e` + `pipefail` aborted the whole job the first time a readiness
# `grep` found no match. Keep `-u` and `pipefail`, drop `-e`.
set -uo pipefail
CONDA=/work/nvme/bdnh/zwang83/etc/profile.d/conda.sh
source "$CONDA"; conda activate bench-pp
export HF_HOME=/work/nvme/bdnh/zwang83/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME
# Node-local /tmp, NOT /work — Ray's plasma_store AF_UNIX socket path must stay under 107 bytes,
# and /work/nvme/bdnh/zwang83/nnbench/slurm/logs/... + session_<ts>/sockets/plasma_store overflows it.
RAY_TMP=/tmp/ray_$SLURM_JOB_ID
mkdir -p "$RAY_TMP"

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head=${nodes[0]}; worker=${nodes[1]}
# Use the CANONICAL routable inter-node IP (getent), NOT `hostname -I` — Delta nodes are multi-homed
# and `hostname -I` returns a non-routable management NIC (172.28.x) first, which Ray's GCS then binds
# to and no other node can reach. getent resolves the fabric IP (141.142.254.x) that NCCL also uses.
head_ip=$(getent hosts "$head" | awk '{print $1}')
worker_ip=$(getent hosts "$worker" | awk '{print $1}')
export RAY_ADDRESS="${head_ip}:6379"
echo "head=$head ($head_ip)  worker=$worker ($worker_ip)  RAY_ADDRESS=$RAY_ADDRESS"

# Ray head + worker, each exposing exactly ONE GPU so PP=2 (world size 2) must span both nodes.
srun --nodes=1 --ntasks=1 -w "$head" --overlap bash -lc \
  "source $CONDA; conda activate bench-pp; CUDA_VISIBLE_DEVICES=0 \
   ray start --head --node-ip-address=$head_ip --port=6379 --num-gpus=1 \
   --temp-dir=$RAY_TMP --include-dashboard=false --block" &

# Readiness gate: wait until the head GCS answers before joining the worker (instead of a fixed sleep).
for i in $(seq 1 30); do
  RAY_ADDRESS="$RAY_ADDRESS" ray status >/dev/null 2>&1 && { echo "head GCS up after ${i}x3s"; break; }
  sleep 3
done

srun --nodes=1 --ntasks=1 -w "$worker" --overlap bash -lc \
  "source $CONDA; conda activate bench-pp; CUDA_VISIBLE_DEVICES=0 \
   ray start --address=$RAY_ADDRESS --node-ip-address=$worker_ip --num-gpus=1 --block" &

# Wait until BOTH nodes (2 GPUs total) are registered before launching the driver.
for i in $(seq 1 30); do
  ngpu=$(RAY_ADDRESS="$RAY_ADDRESS" ray status 2>/dev/null | grep -oE "[0-9]+\.0/[0-9]+\.0 GPU" | grep -oE "/[0-9]+" | tr -d /)
  [ "${ngpu:-0}" = "2" ] && { echo "cluster ready: 2 GPUs across 2 nodes"; break; }
  sleep 3
done
echo "=== ray status ==="; RAY_ADDRESS="$RAY_ADDRESS" ray status || true

# Driver on the head node. CUDA_VISIBLE_DEVICES=1 -> the in-process (1,1) control uses head GPU 1
# (Ray owns head GPU 0). The Ray candidate ignores this pin and uses the cluster (head g0 + worker g0).
# One fresh bench.py process per spec (clean vLLM state between specs); the external Ray cluster
# persists across them. Override the set with PP_SPECS=... if desired.
SPECS="${PP_SPECS:-logit_lens_gpt2 steering_gpt2 ablation_gpt2 activation_patching_gpt2 attention_pattern_gpt2 attribution_patching_gpt2 gen_steering_gpt2}"
srun --nodes=1 --ntasks=1 -w "$head" --overlap bash -lc \
  "source $CONDA; conda activate bench-pp; \
   export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE RAY_ADDRESS=$RAY_ADDRESS CUDA_VISIBLE_DEVICES=1; \
   cd /work/nvme/bdnh/zwang83/nnbench; \
   for spec in $SPECS; do \
     echo \"==================== SPEC: \$spec ====================\"; \
     python scripts/bench.py --spec \$spec --pp 2 --executor ray || echo \"!! spec \$spec exited non-zero\"; \
   done"

echo '=== BENCH DONE — tearing down ray ==='
srun --nodes=1 --ntasks=1 -w "$worker" --overlap bash -lc "source $CONDA; conda activate bench-pp; ray stop" || true
srun --nodes=1 --ntasks=1 -w "$head"   --overlap bash -lc "source $CONDA; conda activate bench-pp; ray stop" || true
# Do NOT `wait` on the backgrounded `ray start --block` srun steps: after `ray stop` they may not
# exit, which hangs the batch script to its wall-clock TIMEOUT (the benchmark itself already finished).
# Kill the background steps and exit explicitly so the job releases the nodes immediately.
kill $(jobs -p) 2>/dev/null || true
echo '=== JOB EXIT CLEAN ==='
exit 0

#!/bin/bash
# Build Apptainer images for the cross-system comparison (docs/delta-handoff.md step 0).
# Usage: sbatch --job-name=isb-build-a slurm/delta-build.slurm BACKEND [BACKEND ...]
# Each backend is built separately so one failure does not block the rest.
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
# NDIF-A40/NDIF-H200 are MAGNETIC for bdnh-delta-gpu; never run on their reserved nodes.
#SBATCH --exclude=gpue01,gpub003,gpub010,gpub025,gpub037,gpub050,gpub069,gpub088
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -uo pipefail
cd /work/nvme/bdnh/zwang83/nnbench
PY=/work/nvme/bdnh/zwang83/envs/isb-host/bin/python
export ISB_LAUNCHER=apptainer
export ISB_APPTAINER_DIR=$PWD/.apptainer
# Per-job layer cache: concurrent build jobs never write the same cache blobs.
export APPTAINER_CACHEDIR=/work/nvme/bdnh/zwang83/.apptainer-cache/${SLURM_JOB_NAME}
export APPTAINER_TMPDIR=/tmp/isb-build-$SLURM_JOB_ID
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT

echo "host=$(hostname) git=$(git rev-parse --short HEAD) apptainer=$(apptainer --version)"
failed=()
for backend in "$@"; do
    start=$(date +%s)
    if $PY scripts/bench.py build "$backend"; then
        echo "[build-ok] $backend $(( $(date +%s) - start ))s"
    else
        echo "[build-FAILED] $backend $(( $(date +%s) - start ))s"
        failed+=("$backend")
    fi
done
ls -la "$ISB_APPTAINER_DIR/images/"
df -h /work/nvme | tail -1
[ ${#failed[@]} -eq 0 ] || { echo "FAILED: ${failed[*]}"; exit 1; }

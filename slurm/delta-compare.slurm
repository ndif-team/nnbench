#!/bin/bash
# Cross-system comparison on Delta via the Apptainer launcher (docs/delta-handoff.md).
# Usage: sbatch [--job-name=NAME] slurm/delta-compare.slurm STAGE [ARGS]
#   smoke             steps 2a-2d; exits nonzero if any job failed
#   rebuild-full      rebuild transformer-lens, re-check its smoke rows, then `full`
#   full              step 3 + the fp32 precision check + HTML export and tarball
#   nnsight-versions  nnsight 0.7.0 vs 0.8.0rc1 on HF and vLLM (smoke gate, then five specs)
#   overhead IMG:hf|vllm ...   per-trace overhead decomposition (slurm/diag_overhead.py)
# Images come from slurm/delta-build.slurm. Runs and logs land under runs/ and slurm/logs/.
#SBATCH --account=bdnh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
# NDIF-A40/NDIF-H200 are MAGNETIC for bdnh-delta-gpu; never run on their reserved nodes.
#SBATCH --exclude=gpue01,gpub003,gpub010,gpub025,gpub037,gpub050,gpub069,gpub088
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -uo pipefail
cd /work/nvme/bdnh/zwang83/nnbench
PY=/work/nvme/bdnh/zwang83/envs/isb-host/bin/python
export ISB_LAUNCHER=apptainer
export ISB_APPTAINER_DIR=$PWD/.apptainer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # model comes from the cache volume

echo "host=$(hostname) git=$(git rev-parse --short HEAD) stage=$1 start=$(date -Is)"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv
git status --short isb backends

bench() { echo; echo "### $(date -Is) bench.py $*"; $PY scripts/bench.py "$@"; local rc=$?; echo "### rc=$rc"; return $rc; }

case "$1" in
smoke)
    out=runs/delta-smoke
    rc=0
    bench run --spec cmp_steering --data counterfact:4 \
        --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0 --out $out || rc=1
    # 2a exercises the launcher end to end; stop here if even that failed.
    [ $rc -eq 0 ] || { echo "2a failed; skipping the rest"; exit 1; }
    bench run --spec cmp_steering --data counterfact:4 \
        --backends nnsight-hf vllm-lens vllm-plain-0-19-1 --reference nnsight-hf --gpu 0 --out $out || rc=1
    bench run --spec cmp_steering cmp_ablation --data counterfact:4 \
        --backends nnsight-hf interp-engine transformer-lens vllm-plain-0-20-2 vllm-plain-0-28-0 \
        --reference nnsight-hf --gpu 0 --out $out || rc=1
    bench run --spec cmp_activation_patching --data mib/ioi:4 \
        --backends nnsight-hf nnsight-vllm vllm-lens interp-engine transformer-lens \
        --reference nnsight-hf --gpu 0 --out $out || rc=1
    exit $rc
    ;;
rebuild-full)
    # Rebuild transformer-lens (transformers pin), re-check its smoke rows, then the full run.
    export APPTAINER_CACHEDIR=/work/nvme/bdnh/zwang83/.apptainer-cache/isb-build-b
    export APPTAINER_TMPDIR=/tmp/isb-build-$SLURM_JOB_ID; mkdir -p $APPTAINER_TMPDIR
    bench build transformer-lens || exit 1
    rm -rf $APPTAINER_TMPDIR
    bench run --spec cmp_steering cmp_ablation --data counterfact:4 \
        --backends nnsight-hf transformer-lens vllm-plain-0-20-2 --reference nnsight-hf \
        --gpu 0 --out runs/delta-smoke || { echo "TL re-check failed; skipping full run"; exit 1; }
    exec bash "$0" full
    ;;
full)
    out=runs/comparison-delta
    rc=0
    bench run \
        --spec cmp_logit_lens cmp_steering cmp_gen_steering cmp_activation_patching cmp_ablation \
        --backends nnsight-hf nnsight-vllm vllm-lens interp-engine transformer-lens \
                   vllm-plain-0-19-1 vllm-plain-0-20-2 vllm-plain-0-28-0 \
        --reference nnsight-hf --gpu 0 --timeout 3600 --out $out || rc=1
    # Optional precision check. 7B fp32 is ~30 GB of weights on a 40 GB card: may not fit.
    bench run --spec cmp_ablation \
        --backends nnsight-hf-fp32 nnsight-vllm-fp32 transformer-lens-fp32 \
        --reference nnsight-hf-fp32 --gpu 0 --timeout 3600 --out $out || echo "fp32 check failed (non-gating)"
    $PY scripts/manager.py --dir $out --export comparison-delta.html
    tar czf comparison-delta.tgz $out
    exit $rc
    ;;
nnsight-versions)
    # nnsight 0.7.0 vs 0.8.0rc1 on HF and vLLM. 0.7 runs on vLLM 0.15.1 (its validated pin),
    # 0.8 on 0.19.1; each has its plain vLLM for overhead_vs_plain_vllm.
    out=runs/nnsight-versions-delta
    bench run --spec cmp_steering --data counterfact:4 \
        --backends nnsight-hf nnsight-hf-0-7 nnsight-vllm-0-7 vllm-plain-0-15-1 --reference nnsight-hf \
        --gpu 0 --out runs/delta-smoke || { echo "0.7 smoke failed; skipping full run"; exit 1; }
    bench run \
        --spec cmp_logit_lens cmp_steering cmp_gen_steering cmp_activation_patching cmp_ablation \
        --backends nnsight-hf nnsight-hf-0-7 nnsight-vllm nnsight-vllm-0-7 \
                   vllm-plain-0-15-1 vllm-plain-0-19-1 \
        --reference nnsight-hf --gpu 0 --timeout 3600 --out $out
    rc=$?
    $PY scripts/manager.py --dir $out --export nnsight-versions-delta.html
    exit $rc
    ;;
overhead)
    # Per-trace overhead decomposition (slurm/diag_overhead.py) in each nnsight image.
    # Usage: sbatch slurm/delta-compare.slurm overhead IMAGE:hf|vllm [...]
    shift
    mkdir -p runs/nnsight-versions-diag/overhead
    rc=0
    for pair in "$@"; do
        img=${pair%%:*}; kind=${pair##*:}
        echo "### $(date -Is) overhead $img"
        timeout 1800 apptainer exec --nv --cleanenv --no-home --pwd /workspace \
            --bind $PWD:/workspace:ro --bind $PWD/runs/nnsight-versions-diag/overhead:/out \
            --bind $ISB_APPTAINER_DIR/volumes/isb-model-cache:/models \
            --env PYTHONPATH=/workspace,PYTHONDONTWRITEBYTECODE=1,USER=isb,HF_HOME=/models,HF_HUB_CACHE=/models/hub \
            --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,XDG_CACHE_HOME=/tmp/cache,VLLM_CACHE_ROOT=/tmp/vllm \
            --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor,TRITON_CACHE_DIR=/tmp/triton \
            --env FLASHINFER_WORKSPACE_BASE=/tmp,CUDA_VISIBLE_DEVICES=0 \
            .apptainer/images/isb-${img}_local.sif python3 slurm/diag_overhead.py $kind /out/$img.json \
            > runs/nnsight-versions-diag/overhead/$img.log 2>&1 || { echo "overhead $img failed rc=$?"; rc=1; }
        sleep 20
    done
    exit $rc
    ;;
*) echo "unknown stage $1"; exit 2 ;;
esac

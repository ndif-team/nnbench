#!/bin/bash
# Cross-system comparison on Delta via the Apptainer launcher (docs/delta-handoff.md).
# Usage: sbatch [--job-name=NAME] slurm/delta-compare.slurm STAGE [ARGS]
#   smoke             steps 2a-2d; exits nonzero if any job failed
#   rebuild-full      rebuild transformer-lens, re-check its smoke rows, then `full`
#   full              step 3 + the fp32 precision check + HTML export and tarball
#   nnsight-versions  nnsight 0.7.0 vs 0.8.0rc1 on HF and vLLM (smoke gate, then five specs)
#   overhead IMG:hf|vllm ...   per-trace overhead decomposition (slurm/diag_overhead.py)
#   fair SPEC ...     the fairness-controlled comparison: every system incl. nnsight 0.7 and the
#                     sync engine, plain vLLM eager and compiled per version, scored vs HF fp32
#   micro             construct probes (nnsight 0.7 / 0.8 x HF / vLLM async / sync) and the
#                     perf microbenchmark sweeps (plans/delta_perf_vllm0191.json, ..._vllm0151.json)
#   perf-read         the perf read cells again (plans/delta_perf_read_*.json)
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

# One python per backend image, as scripts/perf.py's ISB_PY_<system> override expects. Each call
# waits for the GPU to drain afterwards: perf.py starts the next cell's engine at once.
make_wrappers() {
    mkdir -p $bin runs/micro-delta/constructs runs/micro-delta/perf
    for img in vllm-plain-0-19-1 vllm-plain-0-15-1 nnsight-vllm nnsight-vllm-0-7 vllm-lens nnsight-hf nnsight-hf-0-7; do
        cat > $bin/py-$img <<WRAP
#!/bin/bash
apptainer exec --nv --cleanenv --no-home --bind $PWD --bind /tmp \\
    --bind $ISB_APPTAINER_DIR/volumes/isb-model-cache:/models --pwd $PWD \\
    --env PYTHONPATH=$PWD,PYTHONDONTWRITEBYTECODE=1,USER=isb,HF_HOME=/models,HF_HUB_CACHE=/models/hub \\
    --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,XDG_CACHE_HOME=/tmp/cache,VLLM_CACHE_ROOT=/tmp/vllm \\
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor,TRITON_CACHE_DIR=/tmp/triton \\
    --env FLASHINFER_WORKSPACE_BASE=/tmp,CUDA_VISIBLE_DEVICES=0 \\
    $ISB_APPTAINER_DIR/images/isb-\${img}_local.sif python3 "\$@"
rc=\$?
for _ in \$(seq 120); do
    used=\$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "\$used" -lt 1024 ] && break; sleep 1
done
exit \$rc
WRAP
        sed -i "s/\\\${img}/$img/" $bin/py-$img; chmod +x $bin/py-$img
    done
}
bin=runs/micro-delta/bin

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
fair)
    # Fairness-controlled rerun: prefix caching off on every vLLM engine, an eager and a compiled
    # plain vLLM per version (overhead matched to each system's execution mode), nnsight through
    # both its async and sync engines, and HF fp32 as the reference (HF bf16 is itself scored).
    shift
    bench run --spec "$@" \
        --backends nnsight-hf-fp32 nnsight-hf nnsight-hf-0-7 \
                   nnsight-vllm nnsight-vllm-sync nnsight-vllm-0-7 vllm-lens interp-engine transformer-lens \
                   vllm-plain-0-15-1 vllm-plain-0-19-1 vllm-plain-0-20-2 vllm-plain-0-28-0 \
                   vllm-plain-eager-0-15-1 vllm-plain-eager-0-19-1 vllm-plain-eager-0-20-2 vllm-plain-eager-0-28-0 \
        --reference nnsight-hf-fp32 --gpu 0 --timeout 3600 --out runs/fair-delta
    exit $?
    ;;
micro)
    make_wrappers
    rc=0
    # Construct probes on GPT-2: nnsight's primitive map per version and engine interface.
    for pair in nnsight-hf:hf nnsight-hf-0-7:hf nnsight-vllm:vllm_async nnsight-vllm:vllm_sync \
                nnsight-vllm-0-7:vllm_async nnsight-vllm-0-7:vllm_sync; do
        img=${pair%%:*}; be=${pair##*:}
        echo "### $(date -Is) constructs $img $be"
        timeout 2400 $bin/py-$img scripts/micro.py --backend $be --repo openai-community/gpt2 \
            --out runs/micro-delta/constructs --name $img-$be \
            > runs/micro-delta/constructs/$img-$be.log 2>&1 || { echo "constructs $img $be rc=$?"; rc=1; }
    done
    # Perf microbenchmark: one sweep per vLLM version, each with its own pure-vLLM baseline.
    echo "### $(date -Is) perf vLLM 0.19.1"
    ISB_PY_pure_vllm=$bin/py-vllm-plain-0-19-1 ISB_PY_nnsight_vllm=$bin/py-nnsight-vllm \
    ISB_PY_vllm_lens=$bin/py-vllm-lens \
        $PY scripts/perf.py sweep --plan plans/delta_perf_vllm0191.json --name delta_perf_vllm0191 \
        --run-dir runs/micro-delta/perf --out runs/micro-delta/perf/delta_perf_vllm0191.json || rc=1
    echo "### $(date -Is) perf vLLM 0.15.1"
    ISB_PY_pure_vllm=$bin/py-vllm-plain-0-15-1 ISB_PY_nnsight_vllm=$bin/py-nnsight-vllm-0-7 \
        $PY scripts/perf.py sweep --plan plans/delta_perf_vllm0151.json --name delta_perf_vllm0151 \
        --run-dir runs/micro-delta/perf --out runs/micro-delta/perf/delta_perf_vllm0151.json || rc=1
    echo "### $(date -Is) micro done rc=$rc"
    exit $rc
    ;;
perf-read)
    # Rerun of the read cells (and their pure-vLLM baselines) after nnsight's reads were bound to a
    # saved name, so they ship to the client like vllm-lens's; rows record transfer_bytes.
    make_wrappers
    rc=0
    ISB_PY_pure_vllm=$bin/py-vllm-plain-0-19-1 ISB_PY_nnsight_vllm=$bin/py-nnsight-vllm \
    ISB_PY_vllm_lens=$bin/py-vllm-lens \
        $PY scripts/perf.py sweep --plan plans/delta_perf_read_vllm0191.json --name delta_perf_read_vllm0191 \
        --run-dir runs/micro-delta/perf --out runs/micro-delta/perf/delta_perf_read_vllm0191.json || rc=1
    ISB_PY_pure_vllm=$bin/py-vllm-plain-0-15-1 ISB_PY_nnsight_vllm=$bin/py-nnsight-vllm-0-7 \
        $PY scripts/perf.py sweep --plan plans/delta_perf_read_vllm0151.json --name delta_perf_read_vllm0151 \
        --run-dir runs/micro-delta/perf --out runs/micro-delta/perf/delta_perf_read_vllm0151.json || rc=1
    exit $rc
    ;;
*) echo "unknown stage $1"; exit 2 ;;
esac

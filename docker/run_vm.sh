#!/usr/bin/env bash
# Run the VM-style nnbench split. The serve server hosts ONE model, so specs are grouped by model:
# for each model we bring the GPU server up ONCE, run the GPU-less client per spec against it, then
# tear down. Stock the reference run files first (GPU runs) — see the README:
# Legacy standalone serve tool. For the main runner use backends/README.md instead.
# This writes legacy .pt artifacts; score them with scripts/score.py, not the new job scorer.
#
#   ./run_vm.sh                          # every spec below
#   GPU=5 ./run_vm.sh steering_gpt2 ablation_gpt2   # selected specs on GPU 5
set -euo pipefail
cd "$(dirname "$0")"

# spec -> model (the spec's family must match the served model)
declare -A MODELS=(
  [logit_lens_gpt2]="openai-community/gpt2"
  [steering_gpt2]="openai-community/gpt2"
  [ablation_gpt2]="openai-community/gpt2"
  [activation_patching_gpt2]="openai-community/gpt2"
  [attention_pattern_gpt2]="openai-community/gpt2"
  [attribution_patching_gpt2]="openai-community/gpt2"
  [logit_lens_llama]="HuggingFaceTB/SmolLM2-135M-Instruct"
)

SPECS=("$@")
if [ ${#SPECS[@]} -eq 0 ]; then
  SPECS=("${!MODELS[@]}")
fi

# group the requested specs by model so one server serves all same-model specs
declare -A BY_MODEL
for spec in "${SPECS[@]}"; do
  model="${MODELS[$spec]:-}"
  if [ -z "$model" ]; then
    echo "!! no model mapping for spec '$spec' — add it to run_vm.sh" >&2
    continue
  fi
  BY_MODEL[$model]+="$spec "
done

for model in "${!BY_MODEL[@]}"; do
  echo "==================== server model=$model ===================="
  if ! MODEL="$model" CUDA_VISIBLE_DEVICES="${GPU:-0}" docker compose up -d --wait server; then
    echo "!! server failed to become healthy for $model — logs:" >&2
    MODEL="$model" docker compose logs --tail 40 server >&2 || true
    MODEL="$model" docker compose down || true
    continue
  fi
  for spec in ${BY_MODEL[$model]}; do
    echo "-------------------- spec=$spec --------------------"
    # The main name-based runner does not accept this legacy interface.
    mkdir -p ../results/runs
    timeout "${CLIENT_TIMEOUT:-360}" docker compose run --rm -T \
      --volume "$PWD/../results/runs:/outputs" vllm_serve \
      --spec "$spec" --host http://server:6677 --out /outputs --name "$spec-serve" || true
  done
  MODEL="$model" docker compose down
done

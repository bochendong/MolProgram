#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the local base-model path}"
: "${DATA_ROOT:?Set DATA_ROOT to the indexed balanced release}"
: "${RUN_MODE:?Set RUN_MODE to smoke or full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/fresh_balanced/seed_36001}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$RUN_MODE" in
  smoke)
    output="$OUTPUT_ROOT/smoke"
    max_steps="${SMOKE_STEPS:-20}"
    save_steps=1000
    milestones=()
    numerical_guards=(--guard-every-microbatch)
    ;;
  full)
    output="$OUTPUT_ROOT/full"
    max_steps=16283
    save_steps="${SAVE_STEPS:-2500}"
    milestones=(
      --milestone-step 1539
      --milestone-step 3077
      --milestone-step 7693
      --milestone-step 16283
    )
    numerical_guards=()
    ;;
  *)
    echo "RUN_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

resume=()
compgen -G "$output/checkpoint-*" >/dev/null && resume+=(--resume-from-checkpoint)
"$PYTHON_BIN" "$REPO_ROOT/scripts/train_indexed_sft.py" \
  --release-root "$DATA_ROOT" \
  --output-dir "$output" \
  --base-model "$MODEL" \
  --fresh-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --sampler-mode balanced \
  --expected-train-rows 2569919 \
  --max-steps "$max_steps" \
  --per-device-batch-size 1 \
  --gradient-accumulation 65 \
  --learning-rate 2e-5 \
  --warmup-steps 100 \
  --save-steps "$save_steps" \
  --seed 36001 \
  "${milestones[@]}" \
  "${numerical_guards[@]}" \
  "${resume[@]}"

test -f "$output/TRAINING_COMPLETE"
if [[ "$RUN_MODE" == full ]]; then
  for step in 1539 3077 7693 16283; do
    test -f "$output/milestones/checkpoint-$step/adapter/adapter_model.safetensors"
    test -f "$output/milestones/checkpoint-$step/milestone_manifest.json"
  done
fi

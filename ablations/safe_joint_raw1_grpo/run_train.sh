#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL:?Set MODEL to the local backbone path}"
: "${INPUT_ADAPTER:?Set INPUT_ADAPTER to the fresh balanced non-task-aligned adapter}"
: "${TRAIN_JSONL:?Set TRAIN_JSONL to frozen RL training rows}"
: "${SUPPORT_REPORT:?Set SUPPORT_REPORT to the passed editing support report}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/safe-joint-raw1-grpo/seed-37001}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/scripts/train_safe_joint_raw1_grpo.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$MODEL" \
  --input-adapter "$INPUT_ADAPTER" \
  --editing-support-report "$SUPPORT_REPORT" \
  --output-dir "$OUTPUT_ROOT/model/rl" \
  --paired-steps 30 --group-size 16 --learning-rate 1.5e-7 \
  --denovo-anchor-weight 1.5 --edit-anchor-weight 1.5 \
  --reference-kl-weight 0.10 --grad-clip 0.5 \
  --checkpoint-every 10 --seed 37001

"$PYTHON_BIN" "$ROOT/scripts/train_continued_sft_control.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$MODEL" \
  --input-adapter "$INPUT_ADAPTER" \
  --output-dir "$OUTPUT_ROOT/model/continued_sft" \
  --paired-steps 30 --learning-rate 1.5e-7 --grad-clip 0.5 \
  --checkpoint-every 10 --seed 37001

for arm in rl continued_sft; do
  for step in 010 020 030; do
    test -f "$OUTPUT_ROOT/model/$arm/checkpoint-$step/CHECKPOINT_COMPLETE"
  done
done
touch "$OUTPUT_ROOT/TRAIN_COMPLETE"

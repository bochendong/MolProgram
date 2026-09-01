#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the local base-model path}"
: "${INPUT_ADAPTER:?Set INPUT_ADAPTER to the SFT adapter}"
TRAIN_JSONL="${TRAIN_JSONL:-data/rl/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/editing-support-audit}"
PROMPTS_PER_TASK="${PROMPTS_PER_TASK:-50}"
GROUP_SIZE="${GROUP_SIZE:-32}"

python scripts/audit_editing_reward_support.py \
  --train-jsonl "$TRAIN_JSONL" \
  --base-model "$MODEL" \
  --adapter-dir "$INPUT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --prompts-per-task "$PROMPTS_PER_TASK" \
  --group-size "$GROUP_SIZE"

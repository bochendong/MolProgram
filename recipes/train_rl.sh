#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the local or Hugging Face base-model path}"
: "${INPUT_ADAPTER:?Set INPUT_ADAPTER to the SFT adapter}"
TRAIN_JSONL="${TRAIN_JSONL:-data/rl/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl}"

python scripts/train_rl.py \
  --train-jsonl "$TRAIN_JSONL" \
  --base-model "$MODEL" \
  --input-adapter "$INPUT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --rounds 3 \
  --group-size 4 \
  --learning-rate 5e-7 \
  --sft-anchor-weight 1.0 \
  --initial-adapter-weight 0.05

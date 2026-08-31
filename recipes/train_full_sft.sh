#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the local or Hugging Face base-model path}"
: "${INPUT_ADAPTER:?Set INPUT_ADAPTER to the aligned LoRA adapter}"
DATA_ROOT="${DATA_ROOT:-data/molprogram}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/full-sft}"

python scripts/train_indexed_sft.py \
  --release-root "$DATA_ROOT" \
  --base-model "$MODEL" \
  --input-adapter "$INPUT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --sampler-mode proportional_one_pass \
  --num-train-epochs 1 \
  --per-device-batch-size 1 \
  --gradient-accumulation 64

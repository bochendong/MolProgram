#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL:?Set MODEL to the local or Hugging Face base-model path}"
SOURCE="${SOURCE:-data/molprogram}"
PROMPTS="${PROMPTS:-data/eval/denovo.jsonl}"
EDIT_PROMPTS="${EDIT_PROMPTS:-data/eval/edit.jsonl}"
WORK_DIR="${WORK_DIR:-outputs/joint-vs-specialists}"
TRAIN_PER_TASK="${TRAIN_PER_TASK:-10000}"

python ablations/joint_vs_specialists/prepare_data.py \
  --train-source "$SOURCE" \
  --denovo-prompts "$PROMPTS" \
  --edit-prompts "$EDIT_PROMPTS" \
  --output-dir "$WORK_DIR/data" \
  --denovo-train-total "$TRAIN_PER_TASK" \
  --edit-train-total "$TRAIN_PER_TASK"

for arm in joint denovo edit; do
  python ablations/joint_vs_specialists/train_arm.py \
    --train-jsonl "$WORK_DIR/data/train.$arm.jsonl" \
    --base-model "$MODEL" \
    --arm "$arm" \
    --output-dir "$WORK_DIR/$arm"

  python scripts/evaluate_raw1.py \
    --denovo-gate "$WORK_DIR/data/gate.denovo.jsonl" \
    --edit-gate "$WORK_DIR/data/gate.edit.jsonl" \
    --base-model "$MODEL" \
    --adapter-dir "$WORK_DIR/$arm/adapter" \
    --arm "$arm" \
    --output-dir "$WORK_DIR/$arm/eval"
done

python ablations/joint_vs_specialists/collect.py \
  --output-root "$WORK_DIR"

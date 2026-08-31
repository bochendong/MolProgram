#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
RUN_KIND="${MUMO_RUN_KIND:?MUMO_RUN_KIND must be smoke or full}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_JSONL="${MUMO_TRAIN_JSONL:?Set MUMO_TRAIN_JSONL to the indexed release}"
BASE="${MUMO_BASE_MODEL:?Set MUMO_BASE_MODEL to the local backbone path}"
PY="${MUMO_PYTHON_BIN:-python}"
DEP="${MUMO_DEP_OVERLAY:-$REPO_ROOT/src}"
OUT_ROOT="${MUMO_OUTPUT_ROOT:-$REPO_ROOT/outputs/baselines/mumo_fresh/seed_32002}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 cuda/12.6
export PYTHONPATH="$DEP:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=50
  SAVE_STEPS=50
else
  MAX_STEPS=-1
  SAVE_STEPS=600
fi

"$PY" "$SCRIPT_DIR/train.py" \
  --train-jsonl "$TRAIN_JSONL" --base-model "$BASE" \
  --output-dir "$OUT_ROOT/$RUN_KIND" --run-kind "$RUN_KIND" \
  --batch-size 4 --gradient-accumulation 32 --epochs 1 \
  --max-steps "$MAX_STEPS" --learning-rate 2e-5 \
  --save-steps "$SAVE_STEPS" --seed 32002

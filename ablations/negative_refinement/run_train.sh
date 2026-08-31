#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${NEGATIVE_ABLATION_DIR:?NEGATIVE_ABLATION_DIR must be exported}"
ARM="${NEGATIVE_ARM:?NEGATIVE_ARM must be positive_only or semantic_only}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_ROOT="${NEGATIVE_SOURCE_ROOT:?Set NEGATIVE_SOURCE_ROOT to the frozen refinement-data root}"
OUT_ROOT="${NEGATIVE_OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/negative_refinement/seed_2323}"
BASE="${NEGATIVE_BASE_MODEL:?Set NEGATIVE_BASE_MODEL to the local backbone path}"
PY="${NEGATIVE_PYTHON_BIN:-python}"
DEP="${NEGATIVE_DEP_OVERLAY:-$REPO_ROOT/src}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 cuda/12.6
export PYTHONPATH="$DEP:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$PY" "$SCRIPT_DIR/train_refinement.py" \
  --train-jsonl "$SOURCE_ROOT/data/train.contrastive.jsonl" \
  --input-adapter "$SOURCE_ROOT/model/sft/adapter" \
  --base-model "$BASE" \
  --arm "$ARM" \
  --output-dir "$OUT_ROOT/model/$ARM" \
  --epochs 0.25 --gradient-accumulation 16 --learning-rate 1e-5 --seed 2323

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${AUDIT_SCRIPT_DIR:?AUDIT_SCRIPT_DIR must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="${AUDIT_BASE_MODEL:?Set AUDIT_BASE_MODEL to the local backbone path}"
ADAPTER="${AUDIT_ADAPTER_DIR:?Set AUDIT_ADAPTER_DIR to the SFT adapter}"
TRAIN_JSONL="${AUDIT_TRAIN_JSONL:?Set AUDIT_TRAIN_JSONL to the frozen audit rows}"
OUTPUT_ROOT="${AUDIT_OUTPUT_ROOT:?Set AUDIT_OUTPUT_ROOT before submission}"
PY="${AUDIT_PYTHON_BIN:-python}"
DEP="${AUDIT_DEP_OVERLAY:-$REPO_ROOT/src}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PY" "$REPO_ROOT/scripts/audit_editing_reward_support.py" \
  --train-jsonl "$TRAIN_JSONL" \
  --base-model "$BASE" \
  --adapter-dir "$ADAPTER" \
  --output-dir "$OUTPUT_ROOT" \
  --prompts-per-task 50 \
  --group-size 32 \
  --seed 41001

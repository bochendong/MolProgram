#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${FRESH_SCRIPT_DIR:?FRESH_SCRIPT_DIR must be exported}"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_MODE="${FRESH_RUN_MODE:?FRESH_RUN_MODE must be smoke or full}"
PY="${FRESH_PYTHON_BIN:-python}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${FRESH_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${FRESH_HF_CACHE:-$ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="$FRESH_MODEL" DATA_ROOT="$FRESH_DATA_ROOT" RUN_MODE="$RUN_MODE" \
  OUTPUT_ROOT="$FRESH_OUTPUT_ROOT" PYTHON_BIN="$PY" \
  bash "$SCRIPT_DIR/run_train.sh"

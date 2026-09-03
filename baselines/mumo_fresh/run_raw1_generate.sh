#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
PY="${MUMO_RAW1_PYTHON_BIN:-python}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${MUMO_RAW1_DEP_OVERLAY:-$SCRIPT_DIR}:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${MUMO_RAW1_HF_CACHE:-$SCRIPT_DIR/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT_DIR/generate_raw1.py" \
  --rows-csv "$MUMO_RAW1_ROWS_CSV" \
  --source-json "$MUMO_RAW1_SOURCE_JSON" \
  --base-model "$MUMO_RAW1_BASE_MODEL" \
  --adapter-dir "$MUMO_RAW1_ADAPTER" \
  --output-dir "$MUMO_RAW1_OUTPUT_ROOT/generation" \
  --batch-size 1 \
  --seed 32021
test -f "$MUMO_RAW1_OUTPUT_ROOT/generation/GENERATION_COMPLETE"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
PY="${MUMO_RAW1_PYTHON_BIN:-python}"
"$PY" "$SCRIPT_DIR/collect_raw1.py" \
  --summary-csv "$MUMO_RAW1_OUTPUT_ROOT/evaluation/external_multiproperty_summary.csv" \
  --generation-summary "$MUMO_RAW1_OUTPUT_ROOT/generation/generation_summary.json" \
  --output-dir "$MUMO_RAW1_OUTPUT_ROOT/result"

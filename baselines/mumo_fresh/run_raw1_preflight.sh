#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
PY="${MUMO_RAW1_PYTHON_BIN:-python}"
for path in "$MUMO_RAW1_ROWS_CSV" "$MUMO_RAW1_SOURCE_JSON" \
  "$MUMO_RAW1_ADAPTER/adapter_model.safetensors" "$MUMO_RAW1_MERGE_ORACLE"; do
  test -f "$path"
done
test -f "$MUMO_RAW1_EVAL_TOOL_ROOT/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
test -f "$MUMO_RAW1_EVAL_TOOL_ROOT/scripts/evaluate_external_multiproperty_predictions.py"
"$PY" -m py_compile "$SCRIPT_DIR/generate_raw1.py" "$SCRIPT_DIR/collect_raw1.py"
"$PY" -m pytest -q "$(cd "$SCRIPT_DIR/../.." && pwd)/tests/test_mumo_baseline_contract.py"

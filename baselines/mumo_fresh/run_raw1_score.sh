#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
PY="${MUMO_RAW1_PYTHON_BIN:-python}"
ADMET_PY="${MUMO_RAW1_ADMET_PYTHON_BIN:?MUMO_RAW1_ADMET_PYTHON_BIN is required}"
PRED="$MUMO_RAW1_OUTPUT_ROOT/generation/predictions.csv"
ORACLE_DIR="$MUMO_RAW1_OUTPUT_ROOT/oracle"
ORACLE="$ORACLE_DIR/generated_properties.csv"
EVAL_DIR="$MUMO_RAW1_OUTPUT_ROOT/evaluation"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="${MUMO_RAW1_DEP_OVERLAY:-$SCRIPT_DIR}:$MUMO_RAW1_EVAL_TOOL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
test -f "$MUMO_RAW1_OUTPUT_ROOT/generation/GENERATION_COMPLETE"
SUCC_ORACLE_INPUT_CSV="$PRED" \
SUCC_ORACLE_OUTPUT_CSV="$ORACLE" \
SUCC_ORACLE_WORK_DIR="$ORACLE_DIR/work" \
SUCC_ORACLE_MERGE_PROPERTIES_CSV="$MUMO_RAW1_MERGE_ORACLE" \
SUCC_PYTHON_BIN="$PY" \
SUCC_ADMET_PYTHON_BIN="$ADMET_PY" \
bash "$MUMO_RAW1_EVAL_TOOL_ROOT/scripts/run_external_multiproperty_generated_oracle_pipeline.sh"
"$PY" "$MUMO_RAW1_EVAL_TOOL_ROOT/scripts/evaluate_external_multiproperty_predictions.py" \
  --prediction-csv "$PRED" \
  --output-dir "$EVAL_DIR" \
  --generated-properties-csv "$ORACLE" \
  --source-properties-csv "$ORACLE" \
  --group-column condition_id \
  --min-source-tanimoto 0.4 \
  --report-title "Fresh MuMO LoRA official Raw@1"
test -f "$EVAL_DIR/external_multiproperty_summary.csv"

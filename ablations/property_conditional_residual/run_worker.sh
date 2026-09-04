#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?stage is required}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${WORK_DIR:?WORK_DIR must be exported}"
: "${SOURCE_WORK:?SOURCE_WORK must be exported}"

SCRIPT_DIR="$REPO_ROOT/ablations/property_conditional_residual"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-33501}"
PROTOCOL="property_conditional_residual_rank4_pilot_v1"
DATA="$WORK_DIR/data"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${DEP_OVERLAY:+$DEP_OVERLAY:}$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE:-$WORK_DIR/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -n "${ASSAY_ORACLE_DIR:-}" ]]; then
  export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ASSAY_ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
  export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ASSAY_ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
fi

require_model() {
  : "${MODEL:?MODEL must be exported for GPU stages}"
}

evaluate() {
  local adapter="$1"
  local layout="$2"
  local arm="$3"
  local output="$4"
  require_model
  "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_raw1.py" \
    --denovo-gate "$SOURCE_WORK/data/gate.denovo.jsonl" \
    --edit-gate "$SOURCE_WORK/data/gate.edit.jsonl" \
    --base-model "$MODEL" --adapter-dir "$adapter" \
    --routing-config "$layout" --arm "$arm" --output-dir "$output" \
    --batch-size 8 --seed "$((SEED + 50))" --protocol "$PROTOCOL"
}

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" -m pytest -q "$REPO_ROOT/tests/test_property_conditional_residual.py"
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare.py" \
      --source-train "$SOURCE_WORK/data/train.joint.jsonl" \
      --output-dir "$DATA" --expected-rows 1920
    ;;
  train)
    require_model
    "$PYTHON_BIN" "$SCRIPT_DIR/train_residual.py" \
      --train-jsonl "$DATA/train.edit_only.jsonl" --base-model "$MODEL" \
      --shared-adapter-dir "$SOURCE_WORK/dense/adapter" \
      --conditional-layout "$SCRIPT_DIR/conditional_layout.json" \
      --baseline-output-dir "$WORK_DIR/frozen_shared" \
      --output-dir "$WORK_DIR/conditional_residual" \
      --shared-rank 16 --residual-rank 4 --epochs 1.0 \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --expected-rows 1920 --seed "$SEED"
    ;;
  eval_baseline)
    evaluate "$WORK_DIR/frozen_shared/adapter" \
      "$SCRIPT_DIR/conditional_layout.json" frozen_shared \
      "$WORK_DIR/eval_frozen_shared"
    ;;
  eval_conditional)
    evaluate "$WORK_DIR/conditional_residual/adapter" \
      "$SCRIPT_DIR/conditional_layout.json" conditional_residual \
      "$WORK_DIR/eval_conditional_residual"
    ;;
  eval_always_on)
    evaluate "$WORK_DIR/conditional_residual/adapter" \
      "$SCRIPT_DIR/always_on_layout.json" always_on_residual \
      "$WORK_DIR/eval_always_on_residual"
    ;;
  collect)
    "$PYTHON_BIN" "$SCRIPT_DIR/collect.py" \
      --baseline-summary "$WORK_DIR/eval_frozen_shared/summary.json" \
      --conditional-summary "$WORK_DIR/eval_conditional_residual/summary.json" \
      --always-on-summary "$WORK_DIR/eval_always_on_residual/summary.json" \
      --baseline-candidates "$WORK_DIR/eval_frozen_shared/candidates.jsonl" \
      --conditional-candidates "$WORK_DIR/eval_conditional_residual/candidates.jsonl" \
      --always-on-candidates "$WORK_DIR/eval_always_on_residual/candidates.jsonl" \
      --training-summary "$WORK_DIR/conditional_residual/training_summary.json" \
      --output-dir "$WORK_DIR/result"
    ;;
  *)
    printf 'unsupported stage: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac

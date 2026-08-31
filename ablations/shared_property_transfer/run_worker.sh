#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?stage must be prepare, train, evaluate, or collect}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${MODEL:?MODEL must be exported}"
: "${RELEASE_ROOT:?RELEASE_ROOT must be exported}"
: "${BASELINE_ROOT:?BASELINE_ROOT must be exported}"
: "${WORK_DIR:?WORK_DIR must be exported}"

SCRIPT_DIR="$REPO_ROOT/ablations/shared_property_transfer"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-33101}"
PROTOCOL="${PROTOCOL:-shared_property_transfer_v1}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${DEP_OVERLAY:+$DEP_OVERLAY:}$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE:-$WORK_DIR/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$STAGE" in
  prepare)
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_data.py" \
      --baseline-data-dir "$BASELINE_ROOT/data" \
      --train-source "$RELEASE_ROOT/de_novo" \
      --output-dir "$WORK_DIR/data" --replay-total 10000 \
      --seed "$SEED" --protocol "$PROTOCOL"
    ;;
  train)
    "$PYTHON_BIN" "$REPO_ROOT/ablations/joint_vs_specialists/train_arm.py" \
      --train-jsonl "$WORK_DIR/data/train.shared_property_joint.jsonl" \
      --output-dir "$WORK_DIR/model" --base-model "$MODEL" --arm joint \
      --epochs 1.0 --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "$PROTOCOL"
    ;;
  evaluate)
    "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_raw1.py" \
      --denovo-gate "$WORK_DIR/data/gate.denovo.jsonl" \
      --edit-gate "$WORK_DIR/data/gate.edit.jsonl" \
      --base-model "$MODEL" --adapter-dir "$WORK_DIR/model/adapter" \
      --arm shared_property_joint --output-dir "$WORK_DIR/eval" \
      --seed "$((SEED + 50))" --protocol "$PROTOCOL"
    ;;
  collect)
    "$PYTHON_BIN" "$SCRIPT_DIR/collect.py" \
      --candidate-summary "$WORK_DIR/eval/summary.json" \
      --joint-summary "$BASELINE_ROOT/eval/joint/summary.json" \
      --edit-summary "$BASELINE_ROOT/eval/edit/summary.json" \
      --output-dir "$WORK_DIR/result"
    ;;
  *)
    echo "unsupported stage: $STAGE" >&2
    exit 2
    ;;
esac

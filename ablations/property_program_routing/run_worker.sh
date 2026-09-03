#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?stage must be preflight, smoke, train, evaluate, or collect}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${MODEL:?MODEL must be exported}"
: "${BASELINE_ROOT:?BASELINE_ROOT must be exported}"
: "${WORK_DIR:?WORK_DIR must be exported}"

SCRIPT_DIR="$REPO_ROOT/ablations/property_program_routing"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-33101}"
PROTOCOL="property_program_routed_lora_10k_v1"
LAYOUT="$SCRIPT_DIR/routing_layout.json"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${DEP_OVERLAY:+$DEP_OVERLAY:}$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE:-$WORK_DIR/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

first_existing() {
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'none of the expected baseline files exists: %s\n' "$*" >&2
  return 1
}

case "$STAGE" in
  preflight)
    "$PYTHON_BIN" "$SCRIPT_DIR/preflight.py" \
      --baseline-data-dir "$BASELINE_ROOT/data" \
      --routing-config "$LAYOUT" --output-dir "$WORK_DIR/preflight" \
      --protocol "$PROTOCOL"
    ;;
  smoke)
    "$PYTHON_BIN" "$SCRIPT_DIR/train.py" \
      --train-jsonl "$BASELINE_ROOT/data/train.joint.jsonl" \
      --routing-config "$LAYOUT" --output-dir "$WORK_DIR/smoke" \
      --base-model "$MODEL" --max-steps 2 --epochs 1.0 \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "${PROTOCOL}_smoke"
    ;;
  train)
    "$PYTHON_BIN" "$SCRIPT_DIR/train.py" \
      --train-jsonl "$BASELINE_ROOT/data/train.joint.jsonl" \
      --routing-config "$LAYOUT" --output-dir "$WORK_DIR/model" \
      --base-model "$MODEL" --epochs 1.0 \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "$PROTOCOL"
    ;;
  evaluate)
    "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_raw1.py" \
      --denovo-gate "$BASELINE_ROOT/data/gate.denovo.jsonl" \
      --edit-gate "$BASELINE_ROOT/data/gate.edit.jsonl" \
      --base-model "$MODEL" --adapter-dir "$WORK_DIR/model/adapter" \
      --routing-config "$WORK_DIR/model/adapter/program_routing.json" \
      --arm property_program_routed --output-dir "$WORK_DIR/eval" \
      --batch-size 8 --seed "$((SEED + 50))" --protocol "$PROTOCOL"
    ;;
  collect)
    joint_eval="$(first_existing \
      "$BASELINE_ROOT/joint/eval/summary.json" \
      "$BASELINE_ROOT/eval/joint/summary.json")"
    denovo_eval="$(first_existing \
      "$BASELINE_ROOT/denovo/eval/summary.json" \
      "$BASELINE_ROOT/eval/denovo/summary.json")"
    edit_eval="$(first_existing \
      "$BASELINE_ROOT/edit/eval/summary.json" \
      "$BASELINE_ROOT/eval/edit/summary.json")"
    joint_train="$(first_existing \
      "$BASELINE_ROOT/joint/training_summary.json" \
      "$BASELINE_ROOT/model/joint/training_summary.json")"
    "$PYTHON_BIN" "$SCRIPT_DIR/collect.py" \
      --candidate-summary "$WORK_DIR/eval/summary.json" \
      --joint-summary "$joint_eval" --denovo-summary "$denovo_eval" \
      --edit-summary "$edit_eval" \
      --candidate-train "$WORK_DIR/model/training_summary.json" \
      --joint-train "$joint_train" --output-dir "$WORK_DIR/result"
    ;;
  *)
    printf 'unsupported stage: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac

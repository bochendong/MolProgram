#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?stage is required}"
: "${REPO_ROOT:?REPO_ROOT must be exported}"
: "${WORK_DIR:?WORK_DIR must be exported}"

SCRIPT_DIR="$REPO_ROOT/ablations/transfer_aware_program_routing"
HARD_DIR="$REPO_ROOT/ablations/property_program_routing"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-33401}"
EXPECTED_PER_MODE="${EXPECTED_PER_MODE:-3840}"
WARMUP_STEPS="${WARMUP_STEPS:-16}"
PROTOCOL="transfer_aware_program_routing_3840_per_mode_pilot_v1"
DATA="$WORK_DIR/data"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${DEP_OVERLAY:+$DEP_OVERLAY:}$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_CACHE:-$WORK_DIR/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

require_model() {
  : "${MODEL:?MODEL must be exported for GPU stages}"
}

case "$STAGE" in
  prepare)
    : "${TASK_COVERED_TRAIN:?TASK_COVERED_TRAIN must be exported}"
    : "${GATE_ROOT:?GATE_ROOT must be exported}"
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare.py" \
      --task-covered-train "$TASK_COVERED_TRAIN" \
      --gate-root "$GATE_ROOT" --output-dir "$DATA" --seed "$SEED"
    ;;
  warmup)
    require_model
    "$PYTHON_BIN" "$HARD_DIR/train.py" \
      --train-jsonl "$DATA/train.joint.jsonl" \
      --routing-config "$SCRIPT_DIR/dense_layout.json" \
      --output-dir "$WORK_DIR/warmup" --base-model "$MODEL" \
      --max-steps "$WARMUP_STEPS" --epochs 1.0 \
      --expected-per-mode "$EXPECTED_PER_MODE" \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "${PROTOCOL}_dense_warmup"
    ;;
  probe)
    require_model
    "$PYTHON_BIN" "$SCRIPT_DIR/probe_gradients.py" \
      --train-jsonl "$DATA/train.joint.jsonl" --base-model "$MODEL" \
      --adapter-dir "$WORK_DIR/warmup/adapter" \
      --output-dir "$WORK_DIR/probe" --samples-per-node 4 \
      --rank 16 --common-ranks 8 --inactive-floor 0.25 --seed "$SEED"
    ;;
  smoke)
    require_model
    "$PYTHON_BIN" "$HARD_DIR/train.py" \
      --train-jsonl "$DATA/train.joint.jsonl" \
      --routing-config "$WORK_DIR/probe/routing_layout.json" \
      --input-adapter-dir "$WORK_DIR/warmup/adapter" \
      --output-dir "$WORK_DIR/smoke" --base-model "$MODEL" \
      --max-steps 2 --epochs 1.0 --expected-per-mode "$EXPECTED_PER_MODE" \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "${PROTOCOL}_smoke"
    ;;
  train_dense)
    require_model
    "$PYTHON_BIN" "$HARD_DIR/train.py" \
      --train-jsonl "$DATA/train.joint.jsonl" \
      --routing-config "$SCRIPT_DIR/dense_layout.json" \
      --input-adapter-dir "$WORK_DIR/warmup/adapter" \
      --output-dir "$WORK_DIR/dense" --base-model "$MODEL" \
      --epochs 1.0 --expected-per-mode "$EXPECTED_PER_MODE" \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "${PROTOCOL}_matched_dense"
    ;;
  train_routed)
    require_model
    "$PYTHON_BIN" "$HARD_DIR/train.py" \
      --train-jsonl "$DATA/train.joint.jsonl" \
      --routing-config "$WORK_DIR/probe/routing_layout.json" \
      --input-adapter-dir "$WORK_DIR/warmup/adapter" \
      --output-dir "$WORK_DIR/routed" --base-model "$MODEL" \
      --epochs 1.0 --expected-per-mode "$EXPECTED_PER_MODE" \
      --gradient-accumulation 32 --learning-rate 0.00008 \
      --seed "$SEED" --protocol "$PROTOCOL"
    ;;
  eval_dense)
    require_model
    "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_raw1.py" \
      --denovo-gate "$DATA/gate.denovo.jsonl" \
      --edit-gate "$DATA/gate.edit.jsonl" --base-model "$MODEL" \
      --adapter-dir "$WORK_DIR/dense/adapter" --arm matched_dense \
      --output-dir "$WORK_DIR/eval_dense" --batch-size 8 \
      --seed "$((SEED + 50))" --protocol "$PROTOCOL"
    ;;
  eval_routed)
    require_model
    "$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_raw1.py" \
      --denovo-gate "$DATA/gate.denovo.jsonl" \
      --edit-gate "$DATA/gate.edit.jsonl" --base-model "$MODEL" \
      --adapter-dir "$WORK_DIR/routed/adapter" \
      --routing-config "$WORK_DIR/routed/adapter/program_routing.json" \
      --arm transfer_aware --output-dir "$WORK_DIR/eval_routed" \
      --batch-size 8 --seed "$((SEED + 50))" --protocol "$PROTOCOL"
    ;;
  collect)
    "$PYTHON_BIN" "$SCRIPT_DIR/collect.py" \
      --routed-summary "$WORK_DIR/eval_routed/summary.json" \
      --dense-summary "$WORK_DIR/eval_dense/summary.json" \
      --routed-train "$WORK_DIR/routed/training_summary.json" \
      --dense-train "$WORK_DIR/dense/training_summary.json" \
      --transfer-graph "$WORK_DIR/probe/transfer_graph.json" \
      --output-dir "$WORK_DIR/result"
    ;;
  *)
    printf 'unsupported stage: %s\n' "$STAGE" >&2
    exit 2
    ;;
esac

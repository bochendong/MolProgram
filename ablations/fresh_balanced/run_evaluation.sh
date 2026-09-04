#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:?evaluation stage is required}"
: "${FRESH_EVAL_REPO:?FRESH_EVAL_REPO is required}"
: "${FRESH_EVAL_MODEL:?FRESH_EVAL_MODEL is required}"
: "${FRESH_EVAL_TRAIN_ROOT:?FRESH_EVAL_TRAIN_ROOT is required}"
: "${FRESH_EVAL_GATE_DIR:?FRESH_EVAL_GATE_DIR is required}"
: "${FRESH_EVAL_OUTPUT_ROOT:?FRESH_EVAL_OUTPUT_ROOT is required}"

PY="${FRESH_EVAL_PYTHON:-python}"
SCRIPT_DIR="$FRESH_EVAL_REPO/ablations/fresh_balanced"
PROTOCOL="$SCRIPT_DIR/evaluation_protocol.json"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${FRESH_EVAL_DEP_OVERLAY:+$FRESH_EVAL_DEP_OVERLAY:}$FRESH_EVAL_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${FRESH_EVAL_HF_CACHE:-$FRESH_EVAL_OUTPUT_ROOT/hf-cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
if [[ -n "${FRESH_EVAL_ASSAY_ORACLE_DIR:-}" ]]; then
  export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$FRESH_EVAL_ASSAY_ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
  export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$FRESH_EVAL_ASSAY_ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"
fi

case "$STAGE" in
  100k) step=1539 ;;
  200k) step=3077 ;;
  500k) step=7693 ;;
  full) step=16283 ;;
  collect)
    "$PY" "$SCRIPT_DIR/collect_evaluations.py" \
      --evaluation-root "$FRESH_EVAL_OUTPUT_ROOT/evaluations" \
      --training-root "$FRESH_EVAL_TRAIN_ROOT" \
      --gate-dir "$FRESH_EVAL_GATE_DIR" --protocol "$PROTOCOL" \
      --output-dir "$FRESH_EVAL_OUTPUT_ROOT/result"
    if [[ -n "${FRESH_EVAL_SAFE_GRPO_JOB:-}" ]] && \
      "$PY" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["decision"]["safe_grpo_allowed"])' \
        "$FRESH_EVAL_OUTPUT_ROOT/result/result.json"; then
      scontrol release "$FRESH_EVAL_SAFE_GRPO_JOB"
      printf '%s\n' "$FRESH_EVAL_SAFE_GRPO_JOB" \
        > "$FRESH_EVAL_OUTPUT_ROOT/result/SAFE_GRPO_RELEASED"
    fi
    exit 0
    ;;
  *) echo "unsupported evaluation stage: $STAGE" >&2; exit 2 ;;
esac

adapter="$FRESH_EVAL_TRAIN_ROOT/full/milestones/checkpoint-$step/adapter"
test -f "$adapter/adapter_model.safetensors"
test -f "$adapter/../milestone_manifest.json"
test -f "$FRESH_EVAL_GATE_DIR/manifest.json"
"$PY" -c 'import hashlib,json,sys,pathlib; d=pathlib.Path(sys.argv[1]); m=json.load(open(d/"manifest.json")); assert hashlib.sha256((d/"gate.denovo.jsonl").read_bytes()).hexdigest()==m["de_novo"]["sha256"]; assert hashlib.sha256((d/"gate.edit.jsonl").read_bytes()).hexdigest()==m["editing"]["sha256"]' \
  "$FRESH_EVAL_GATE_DIR"
"$PY" "$FRESH_EVAL_REPO/scripts/evaluate_raw1.py" \
  --denovo-gate "$FRESH_EVAL_GATE_DIR/gate.denovo.jsonl" \
  --edit-gate "$FRESH_EVAL_GATE_DIR/gate.edit.jsonl" \
  --base-model "$FRESH_EVAL_MODEL" --adapter-dir "$adapter" \
  --output-dir "$FRESH_EVAL_OUTPUT_ROOT/evaluations/$STAGE" \
  --arm "fresh_balanced_$STAGE" --batch-size 8 --seed 36051 \
  --protocol fresh_balanced_headline_raw1_v1

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAPTER="${1:?adapter path is required}"
LABEL="${2:?evaluation label is required}"
DENOVO="${3:?de novo gate is required}"
EDIT="${4:?editing gate is required}"
OUT="${5:?output path is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"

test -f "$ADAPTER/adapter_model.safetensors"
"$PY" "$ROOT/scripts/evaluate_raw1.py" \
  --denovo-gate "$DENOVO" --edit-gate "$EDIT" \
  --base-model "$SAFE_GRPO_BASE_MODEL" --adapter-dir "$ADAPTER" \
  --output-dir "$OUT" --arm "$LABEL" --batch-size 8 --seed 37051 \
  --protocol molprogram_safe_joint_raw1_grpo_gate_v1

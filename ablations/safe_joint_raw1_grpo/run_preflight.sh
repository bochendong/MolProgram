#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${SAFE_GRPO_INPUT_ADAPTER:?SAFE_GRPO_INPUT_ADAPTER is required}"
: "${SAFE_GRPO_INPUT_MARKER:?SAFE_GRPO_INPUT_MARKER is required}"
: "${SAFE_GRPO_TRAIN_JSONL:?SAFE_GRPO_TRAIN_JSONL is required}"
: "${SAFE_GRPO_DENOVO_DEV:?SAFE_GRPO_DENOVO_DEV is required}"
: "${SAFE_GRPO_EDIT_DEV:?SAFE_GRPO_EDIT_DEV is required}"
: "${SAFE_GRPO_DENOVO_FINAL:?SAFE_GRPO_DENOVO_FINAL is required}"
: "${SAFE_GRPO_EDIT_FINAL:?SAFE_GRPO_EDIT_FINAL is required}"
: "${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="${SAFE_GRPO_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for path in \
  "$SAFE_GRPO_INPUT_MARKER" \
  "$SAFE_GRPO_INPUT_ADAPTER/adapter_model.safetensors" \
  "$SAFE_GRPO_TRAIN_JSONL" \
  "$SAFE_GRPO_DENOVO_DEV" "$SAFE_GRPO_EDIT_DEV" \
  "$SAFE_GRPO_DENOVO_FINAL" "$SAFE_GRPO_EDIT_FINAL"; do
  test -e "$path"
done

"$PY" -m py_compile \
  "$ROOT/src/molprogram/safe_grpo.py" \
  "$ROOT/scripts/train_safe_joint_raw1_grpo.py" \
  "$ROOT/scripts/train_continued_sft_control.py" \
  "$SCRIPT_DIR/validate_inputs.py" \
  "$SCRIPT_DIR/select_checkpoint.py" "$SCRIPT_DIR/collect.py"
"$PY" -m pytest -q "$ROOT/tests/test_safe_joint_raw1_grpo.py"
"$PY" "$SCRIPT_DIR/validate_inputs.py" \
  --train-jsonl "$SAFE_GRPO_TRAIN_JSONL" \
  --denovo-dev "$SAFE_GRPO_DENOVO_DEV" --edit-dev "$SAFE_GRPO_EDIT_DEV" \
  --denovo-final "$SAFE_GRPO_DENOVO_FINAL" --edit-final "$SAFE_GRPO_EDIT_FINAL" \
  --input-adapter "$SAFE_GRPO_INPUT_ADAPTER" \
  --input-marker "$SAFE_GRPO_INPUT_MARKER" \
  --output "$SAFE_GRPO_OUTPUT_ROOT/input_validation.json"

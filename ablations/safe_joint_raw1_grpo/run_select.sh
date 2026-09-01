#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"
test -f "$OUT/DEV_EVAL_COMPLETE"
for arm in rl continued_sft; do
  "$PY" "$SCRIPT_DIR/select_checkpoint.py" \
    --arm "$arm" --baseline-summary "$OUT/gate/dev/baseline/summary.json" \
    --evaluation-root "$OUT/gate/dev/$arm" --model-root "$OUT/model/$arm" \
    --output "$OUT/gate/dev/${arm}_selection.json" --steps 10 20 30
done
touch "$OUT/DEV_SELECTION_COMPLETE"

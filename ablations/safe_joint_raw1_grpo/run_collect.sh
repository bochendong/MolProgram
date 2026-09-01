#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"
test -f "$OUT/FINAL_EVAL_COMPLETE"
"$PY" "$SCRIPT_DIR/collect.py" \
  --baseline-dir "$OUT/gate/final/baseline" \
  --control-dir "$OUT/gate/final/continued_sft" \
  --rl-dir "$OUT/gate/final/rl" \
  --rl-selection "$OUT/gate/dev/rl_selection.json" \
  --control-selection "$OUT/gate/dev/continued_sft_selection.json" \
  --output-dir "$OUT/result" --bootstrap-replicates 2000 --seed 37101
touch "$OUT/EXPERIMENT_COMPLETE"

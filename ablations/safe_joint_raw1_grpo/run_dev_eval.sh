#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${SAFE_GRPO_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$SCRIPT_DIR/run_one_eval.sh" "$SAFE_GRPO_INPUT_ADAPTER" baseline \
  "$SAFE_GRPO_DENOVO_DEV" "$SAFE_GRPO_EDIT_DEV" "$OUT/gate/dev/baseline"
for arm in rl continued_sft; do
  for step in 010 020 030; do
    "$SCRIPT_DIR/run_one_eval.sh" "$OUT/model/$arm/checkpoint-$step/adapter" \
      "$arm-step$step" "$SAFE_GRPO_DENOVO_DEV" "$SAFE_GRPO_EDIT_DEV" \
      "$OUT/gate/dev/$arm/step$step"
  done
done
touch "$OUT/DEV_EVAL_COMPLETE"

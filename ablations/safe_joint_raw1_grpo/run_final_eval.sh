#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SAFE_GRPO_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROOT="${SAFE_GRPO_REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT="${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"
test -f "$OUT/DEV_SELECTION_COMPLETE"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${SAFE_GRPO_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="$SAFE_GRPO_ASSAY_ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl"
export SUCC_DRD2_ORACLE_PATH="$SAFE_GRPO_ASSAY_ORACLE_DIR/drd2_graph2graph_svc_py36.pkl"

"$SCRIPT_DIR/run_one_eval.sh" "$SAFE_GRPO_INPUT_ADAPTER" baseline \
  "$SAFE_GRPO_DENOVO_FINAL" "$SAFE_GRPO_EDIT_FINAL" "$OUT/gate/final/baseline"
for arm in rl continued_sft; do
  selection="$OUT/gate/dev/${arm}_selection.json"
  adapter="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_adapter"])' "$selection")"
  "$SCRIPT_DIR/run_one_eval.sh" "$adapter" "$arm-selected" \
    "$SAFE_GRPO_DENOVO_FINAL" "$SAFE_GRPO_EDIT_FINAL" "$OUT/gate/final/$arm"
done
touch "$OUT/FINAL_EVAL_COMPLETE"

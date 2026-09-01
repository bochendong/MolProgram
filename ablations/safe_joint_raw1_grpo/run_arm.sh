#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARM="${SAFE_GRPO_ARM:?SAFE_GRPO_ARM must be rl or continued_sft}"
OUT="${SAFE_GRPO_OUTPUT_ROOT:?SAFE_GRPO_OUTPUT_ROOT is required}"
PY="${SAFE_GRPO_PYTHON_BIN:-python}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="${SAFE_GRPO_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SUCC_GSK3B_ORACLE_PATH="$SAFE_GRPO_ASSAY_ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl"
export SUCC_DRD2_ORACLE_PATH="$SAFE_GRPO_ASSAY_ORACLE_DIR/drd2_graph2graph_svc_py36.pkl"

if [[ "$ARM" == rl ]]; then
  test -f "$SAFE_GRPO_SUPPORT_REPORT"
  "$PY" "$ROOT/scripts/train_safe_joint_raw1_grpo.py" \
    --train-jsonl "$SAFE_GRPO_TRAIN_JSONL" --base-model "$SAFE_GRPO_BASE_MODEL" \
    --input-adapter "$SAFE_GRPO_INPUT_ADAPTER" \
    --editing-support-report "$SAFE_GRPO_SUPPORT_REPORT" \
    --output-dir "$OUT/model/rl" \
    --paired-steps 30 --group-size 16 --learning-rate 1.5e-7 \
    --denovo-anchor-weight 1.5 --edit-anchor-weight 1.5 \
    --reference-kl-weight 0.10 --grad-clip 0.5 \
    --checkpoint-every 10 --seed 37001
elif [[ "$ARM" == continued_sft ]]; then
  "$PY" "$ROOT/scripts/train_continued_sft_control.py" \
    --train-jsonl "$SAFE_GRPO_TRAIN_JSONL" --base-model "$SAFE_GRPO_BASE_MODEL" \
    --input-adapter "$SAFE_GRPO_INPUT_ADAPTER" \
    --output-dir "$OUT/model/continued_sft" \
    --paired-steps 30 --learning-rate 1.5e-7 --grad-clip 0.5 \
    --checkpoint-every 10 --seed 37001
else
  echo "SAFE_GRPO_ARM must be rl or continued_sft" >&2
  exit 2
fi
for step in 010 020 030; do
  test -f "$OUT/model/$ARM/checkpoint-$step/CHECKPOINT_COMPLETE"
done
touch "$OUT/${ARM}_TRAIN_COMPLETE"

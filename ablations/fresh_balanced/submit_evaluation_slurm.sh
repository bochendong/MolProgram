#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${FRESH_EVAL_MODEL:?Set FRESH_EVAL_MODEL to the node-local backbone}"
: "${FRESH_EVAL_TRAIN_ROOT:?Set FRESH_EVAL_TRAIN_ROOT to the stable fresh output root}"
: "${FRESH_EVAL_GATE_DIR:?Set FRESH_EVAL_GATE_DIR to the frozen 440/5000 gate directory}"
: "${FRESH_EVAL_TRAIN_JOB:?Set FRESH_EVAL_TRAIN_JOB to the fresh full Slurm job ID}"

ACCOUNT="${FRESH_EVAL_ACCOUNT:-def-hup-ab}"
GPU="${FRESH_EVAL_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT="${FRESH_EVAL_OUTPUT_ROOT:-$FRESH_EVAL_TRAIN_ROOT/headline_raw1}"
LOG_DIR="${FRESH_EVAL_LOG_DIR:-$ROOT/logs/fresh_balanced_evaluation}"
NICE="${FRESH_EVAL_NICE:-1000}"
mkdir -p "$OUT" "$LOG_DIR"

common="FRESH_EVAL_REPO=$ROOT,FRESH_EVAL_MODEL=$FRESH_EVAL_MODEL,FRESH_EVAL_TRAIN_ROOT=$FRESH_EVAL_TRAIN_ROOT,FRESH_EVAL_GATE_DIR=$FRESH_EVAL_GATE_DIR,FRESH_EVAL_OUTPUT_ROOT=$OUT"
for optional in FRESH_EVAL_PYTHON FRESH_EVAL_DEP_OVERLAY FRESH_EVAL_HF_CACHE FRESH_EVAL_ASSAY_ORACLE_DIR FRESH_EVAL_SAFE_GRPO_JOB SUCC_GSK3B_ORACLE_PATH SUCC_DRD2_ORACLE_PATH; do
  [[ -z "${!optional:-}" ]] || common+=",$optional=${!optional}"
done

jobs=()
for label in 100k 200k 500k full; do
  job=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
    --job-name="fresh-eval-$label" --time=01:30:00 --cpus-per-task=4 \
    --mem=40G --gres="$GPU" --dependency="afterok:$FRESH_EVAL_TRAIN_JOB" \
    --kill-on-invalid-dep=yes --output="$LOG_DIR/$label-%j.log" \
    --export="ALL,$common" "$SCRIPT_DIR/run_evaluation.sh" "$label")
  jobs+=("$job")
done

dependency=$(IFS=:; echo "${jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=fresh-eval-collect \
  --time=00:15:00 --cpus-per-task=2 --mem=12G \
  --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/collect-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_evaluation.sh" collect)

printf 'eval_100k=%s eval_200k=%s eval_500k=%s eval_full=%s collect=%s output=%s\n' \
  "${jobs[0]}" "${jobs[1]}" "${jobs[2]}" "${jobs[3]}" "$collect" "$OUT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${NEGATIVE_SOURCE_ROOT:?Set NEGATIVE_SOURCE_ROOT before submission}"
: "${NEGATIVE_GATE_ROOT:?Set NEGATIVE_GATE_ROOT before submission}"
: "${NEGATIVE_BASE_MODEL:?Set NEGATIVE_BASE_MODEL before submission}"
: "${NEGATIVE_ASSAY_ORACLE_DIR:?Set NEGATIVE_ASSAY_ORACLE_DIR before submission}"
ACCOUNT="${NEGATIVE_ACCOUNT:-def-hup-ab}"
GPU="${NEGATIVE_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT_ROOT="${NEGATIVE_OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/negative_refinement/seed_2323}"
LOG_DIR="$REPO_ROOT/logs/ablations/negative_refinement"
mkdir -p "$LOG_DIR"

common="NEGATIVE_ABLATION_DIR=$SCRIPT_DIR,NEGATIVE_OUTPUT_ROOT=$OUT_ROOT"
preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=neg-check \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --output="$LOG_DIR/preflight-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_preflight.sh")

eval_jobs=()
for arm in positive_only semantic_only; do
  short="${arm%%_*}"
  train=$(sbatch --parsable --account="$ACCOUNT" --job-name="neg-t-$short" \
    --time=12:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
    --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/train-$arm-%j.log" \
    --export="ALL,$common,NEGATIVE_ARM=$arm" "$SCRIPT_DIR/run_train.sh")
  eval=$(sbatch --parsable --account="$ACCOUNT" --job-name="neg-e-$short" \
    --time=01:30:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
    --dependency="afterok:$train" --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/eval-$arm-%j.log" \
    --export="ALL,$common,NEGATIVE_ARM=$arm" "$SCRIPT_DIR/run_eval.sh")
  eval_jobs+=("$eval")
  printf 'arm=%s train=%s eval=%s\n' "$arm" "$train" "$eval"
done

all_eval=$(sbatch --parsable --account="$ACCOUNT" --job-name=neg-e-all \
  --time=01:30:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/eval-semantic-plus-syntax-%j.log" \
  --export="ALL,$common,NEGATIVE_ARM=semantic_plus_syntax" "$SCRIPT_DIR/run_eval.sh")
eval_jobs+=("$all_eval")

eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=neg-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$eval_dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/collect-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_collect.sh")

printf 'preflight=%s existing_all_eval=%s collect=%s output=%s\n' \
  "$preflight" "$all_eval" "$collect" "$OUT_ROOT"

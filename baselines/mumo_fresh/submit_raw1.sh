#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
for name in MUMO_RAW1_ROWS_CSV MUMO_RAW1_SOURCE_JSON MUMO_RAW1_BASE_MODEL \
  MUMO_RAW1_ADAPTER MUMO_RAW1_EVAL_TOOL_ROOT MUMO_RAW1_MERGE_ORACLE \
  MUMO_RAW1_PYTHON_BIN MUMO_RAW1_ADMET_PYTHON_BIN; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
ACCOUNT="${MUMO_RAW1_ACCOUNT:-def-hup-ab}"
GPU="${MUMO_RAW1_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT="${MUMO_RAW1_OUTPUT_ROOT:-$ROOT/outputs/baselines/mumo_fresh/stable_v2_seed_32002/raw1}"
LOG_DIR="${MUMO_RAW1_LOG_DIR:-$ROOT/logs/baselines/mumo_fresh/raw1}"
mkdir -p "$OUT" "$LOG_DIR"
dependency_args=()
[[ -n "${MUMO_RAW1_DEPENDENCY:-}" ]] && dependency_args+=(
  --dependency="afterok:$MUMO_RAW1_DEPENDENCY" --kill-on-invalid-dep=yes
)
common="ALL,MUMO_BASELINE_DIR=$SCRIPT_DIR,MUMO_RAW1_OUTPUT_ROOT=$OUT,MUMO_RAW1_ROWS_CSV=$MUMO_RAW1_ROWS_CSV,MUMO_RAW1_SOURCE_JSON=$MUMO_RAW1_SOURCE_JSON,MUMO_RAW1_BASE_MODEL=$MUMO_RAW1_BASE_MODEL,MUMO_RAW1_ADAPTER=$MUMO_RAW1_ADAPTER,MUMO_RAW1_EVAL_TOOL_ROOT=$MUMO_RAW1_EVAL_TOOL_ROOT,MUMO_RAW1_MERGE_ORACLE=$MUMO_RAW1_MERGE_ORACLE,MUMO_RAW1_PYTHON_BIN=$MUMO_RAW1_PYTHON_BIN,MUMO_RAW1_ADMET_PYTHON_BIN=$MUMO_RAW1_ADMET_PYTHON_BIN"
for optional in MUMO_RAW1_DEP_OVERLAY MUMO_RAW1_HF_CACHE; do
  [[ -z "${!optional:-}" ]] || common+=",$optional=${!optional}"
done
preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-raw1-check \
  --time=00:15:00 --cpus-per-task=2 --mem=8G "${dependency_args[@]}" \
  --output="$LOG_DIR/preflight-%j.log" --export="$common" \
  "$SCRIPT_DIR/run_raw1_preflight.sh")
generate=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-raw1-generate \
  --time=04:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/generate-%j.log" --export="$common" \
  "$SCRIPT_DIR/run_raw1_generate.sh")
score=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-raw1-score \
  --time=04:00:00 --cpus-per-task=8 --mem=96G \
  --dependency="afterok:$generate" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/score-%j.log" --export="$common" \
  "$SCRIPT_DIR/run_raw1_score.sh")
collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-raw1-collect \
  --time=00:15:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$score" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/collect-%j.log" --export="$common" \
  "$SCRIPT_DIR/run_raw1_collect.sh")
printf 'preflight=%s\ngenerate=%s\nscore=%s\ncollect=%s\noutput=%s\n' \
  "$preflight" "$generate" "$score" "$collect" "$OUT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${FRESH_MODEL:?Set FRESH_MODEL to the local backbone path}"
: "${FRESH_DATA_ROOT:?Set FRESH_DATA_ROOT to the frozen indexed release}"

ACCOUNT="${FRESH_ACCOUNT:-def-hup-ab}"
GPU="${FRESH_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT="${FRESH_OUTPUT_ROOT:-$ROOT/outputs/fresh_balanced/stable_v2_seed_36001}"
LOG_DIR="${FRESH_LOG_DIR:-$ROOT/logs/fresh_balanced}"
mkdir -p "$OUT" "$LOG_DIR"

begin_args=()
[[ -n "${FRESH_BEGIN:-}" ]] && begin_args+=(--begin="$FRESH_BEGIN")
dependency_args=()
[[ -n "${FRESH_DEPENDENCY:-}" ]] && dependency_args+=(
  --dependency="afterok:$FRESH_DEPENDENCY" --kill-on-invalid-dep=yes
)
common="ALL,FRESH_SCRIPT_DIR=$SCRIPT_DIR,FRESH_MODEL=$FRESH_MODEL,FRESH_DATA_ROOT=$FRESH_DATA_ROOT,FRESH_OUTPUT_ROOT=$OUT"
for optional in FRESH_PYTHON_BIN FRESH_DEP_OVERLAY FRESH_HF_CACHE; do
  [[ -z "${!optional:-}" ]] || common+=",$optional=${!optional}"
done

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=fresh-balanced-check \
  --time=00:15:00 --cpus-per-task=2 --mem=8G \
  "${begin_args[@]}" "${dependency_args[@]}" \
  --output="$LOG_DIR/preflight-%j.log" --export="$common" \
  "$SCRIPT_DIR/run_preflight.sh")

smoke=$(sbatch --parsable --account="$ACCOUNT" --job-name=fresh-balanced-smoke \
  --time=01:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/smoke-%j.log" --export="$common,FRESH_RUN_MODE=smoke" \
  "$SCRIPT_DIR/run_slurm.sh")

full=$(sbatch --parsable --account="$ACCOUNT" --job-name=fresh-balanced-full \
  --time=3-00:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/full-%j.log" --export="$common,FRESH_RUN_MODE=full" \
  "$SCRIPT_DIR/run_slurm.sh")

printf 'preflight_job=%s\nsmoke_job=%s\nfull_job=%s\n' "$preflight" "$smoke" "$full"
printf 'output=%s\nfinal_adapter=%s\ncompletion_marker=%s\n' \
  "$OUT" "$OUT/full/milestones/checkpoint-16283/adapter" \
  "$OUT/full/TRAINING_COMPLETE"

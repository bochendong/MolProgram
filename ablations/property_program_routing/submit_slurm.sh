#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${MODEL:?Set MODEL to a node-local backbone path}"
: "${BASELINE_ROOT:?Set BASELINE_ROOT to the completed matched 10k output}"

ACCOUNT="${ACCOUNT:-def-hup-ab}"
GPU="${GPU:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/property-program-routing-10k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/property-program-routing-10k}"
SEED="${SEED:-33101}"
mkdir -p "$WORK_DIR" "$LOG_DIR"

common="REPO_ROOT=$REPO_ROOT,MODEL=$MODEL,BASELINE_ROOT=$BASELINE_ROOT,WORK_DIR=$WORK_DIR,SEED=$SEED"
for optional in PYTHON_BIN DEP_OVERLAY HF_CACHE; do
  if [[ -n "${!optional:-}" ]]; then
    common+=",$optional=${!optional}"
  fi
done

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-data \
  --time=00:15:00 --cpus-per-task=2 --mem=12G \
  --output="$LOG_DIR/preflight-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" preflight)

smoke=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-smoke \
  --time=00:30:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$preflight" --output="$LOG_DIR/smoke-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" smoke)

train=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-train \
  --time=04:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$smoke" --output="$LOG_DIR/train-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" train)

evaluate=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-eval \
  --time=01:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$train" --output="$LOG_DIR/eval-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" evaluate)

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$evaluate" --output="$LOG_DIR/collect-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" collect)

printf 'preflight=%s smoke=%s train=%s evaluate=%s collect=%s output=%s\n' \
  "$preflight" "$smoke" "$train" "$evaluate" "$collect" "$WORK_DIR"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${MODEL:?Set MODEL to a node-local backbone path}"
: "${RELEASE_ROOT:?Set RELEASE_ROOT to the MolProgram release directory}"
: "${BASELINE_ROOT:?Set BASELINE_ROOT to the matched 10k baseline output}"

ACCOUNT="${ACCOUNT:-def-hup-ab}"
GPU="${GPU:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/shared-property-transfer}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/shared-property-transfer}"
SEED="${SEED:-33101}"
PROTOCOL="shared_property_transfer_v1"
mkdir -p "$WORK_DIR" "$LOG_DIR"

common="REPO_ROOT=$REPO_ROOT,MODEL=$MODEL,RELEASE_ROOT=$RELEASE_ROOT,BASELINE_ROOT=$BASELINE_ROOT,WORK_DIR=$WORK_DIR,SEED=$SEED,PROTOCOL=$PROTOCOL"
for optional in PYTHON_BIN DEP_OVERLAY HF_CACHE; do
  if [[ -n "${!optional:-}" ]]; then
    common+=",$optional=${!optional}"
  fi
done

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-data \
  --time=00:30:00 --cpus-per-task=2 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" prepare)

train=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-train \
  --time=04:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$prepare" --output="$LOG_DIR/train-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" train)

evaluate=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-eval \
  --time=01:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$train" --output="$LOG_DIR/eval-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" evaluate)

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$evaluate" --output="$LOG_DIR/collect-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" collect)

printf 'prepare=%s train=%s evaluate=%s collect=%s output=%s\n' \
  "$prepare" "$train" "$evaluate" "$collect" "$WORK_DIR"

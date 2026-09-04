#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${MODEL:?Set MODEL to the node-local backbone path}"
: "${SOURCE_WORK:?Set SOURCE_WORK to the completed transfer-aware pilot output}"

ACCOUNT="${ACCOUNT:-def-hup-ab}"
GPU="${GPU:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/property-conditional-residual}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/property-conditional-residual}"
SEED="${SEED:-33501}"
NICE="${NICE:-10000}"
mkdir -p "$WORK_DIR" "$LOG_DIR"

common="REPO_ROOT=$REPO_ROOT,MODEL=$MODEL,SOURCE_WORK=$SOURCE_WORK,WORK_DIR=$WORK_DIR,SEED=$SEED"
for optional in PYTHON_BIN DEP_OVERLAY HF_CACHE ASSAY_ORACLE_DIR SUCC_GSK3B_ORACLE_PATH SUCC_DRD2_ORACLE_PATH; do
  if [[ -n "${!optional:-}" ]]; then
    common+=",$optional=${!optional}"
  fi
done

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=pcr-prepare \
  --time=00:15:00 --cpus-per-task=2 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" prepare)

train=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=pcr-train --time=01:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$prepare" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/train-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" train)

eval_baseline=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=pcr-eval-base --time=00:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$train" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/eval-base-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" eval_baseline)

eval_conditional=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=pcr-eval-cond --time=00:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$train" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/eval-cond-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" eval_conditional)

eval_always=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=pcr-eval-always --time=00:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$train" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/eval-always-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" eval_always_on)

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=pcr-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$eval_baseline:$eval_conditional:$eval_always" \
  --kill-on-invalid-dep=yes --output="$LOG_DIR/collect-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" collect)

printf 'prepare=%s train=%s eval_baseline=%s eval_conditional=%s eval_always=%s collect=%s output=%s nice=%s\n' \
  "$prepare" "$train" "$eval_baseline" "$eval_conditional" \
  "$eval_always" "$collect" "$WORK_DIR" "$NICE"

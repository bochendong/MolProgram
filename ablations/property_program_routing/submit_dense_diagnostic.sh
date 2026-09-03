#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${MODEL:?Set MODEL to a node-local backbone path}"
: "${BASELINE_ROOT:?Set BASELINE_ROOT to the completed fresh 10k output}"

ACCOUNT="${ACCOUNT:-def-hup-ab}"
GPU="${GPU:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/property-program-routing-10k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/property-program-routing-10k}"
SEED="${SEED:-33101}"
mkdir -p "$WORK_DIR" "$LOG_DIR"
test -f "$WORK_DIR/model/adapter/adapter_model.safetensors"
test -f "$WORK_DIR/eval/summary.json"

common="REPO_ROOT=$REPO_ROOT,MODEL=$MODEL,BASELINE_ROOT=$BASELINE_ROOT,WORK_DIR=$WORK_DIR,SEED=$SEED"
for optional in PYTHON_BIN DEP_OVERLAY HF_CACHE; do
  if [[ -n "${!optional:-}" ]]; then
    common+=",$optional=${!optional}"
  fi
done

evaluate=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-dense-eval \
  --time=01:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --output="$LOG_DIR/dense-eval-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" evaluate_dense)

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=ppr-dense-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$evaluate" --output="$LOG_DIR/dense-collect-%j.log" \
  --export="ALL,$common" "$SCRIPT_DIR/run_worker.sh" collect_dense)

printf 'dense_eval=%s dense_collect=%s output=%s\n' \
  "$evaluate" "$collect" "$WORK_DIR/dense_diagnostic"

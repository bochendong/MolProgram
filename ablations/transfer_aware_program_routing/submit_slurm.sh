#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${MODEL:?Set MODEL to a node-local backbone path}"
: "${TASK_COVERED_TRAIN:?Set TASK_COVERED_TRAIN to the aligned training JSONL}"
: "${GATE_ROOT:?Set GATE_ROOT to the completed fresh 10k output}"

ACCOUNT="${ACCOUNT:-def-hup-ab}"
GPU="${GPU:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/outputs/transfer-aware-program-routing-10k}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/transfer-aware-program-routing-10k}"
SEED="${SEED:-33401}"
NICE="${NICE:-10000}"
mkdir -p "$WORK_DIR" "$LOG_DIR"

common="REPO_ROOT=$REPO_ROOT,MODEL=$MODEL,TASK_COVERED_TRAIN=$TASK_COVERED_TRAIN,GATE_ROOT=$GATE_ROOT,WORK_DIR=$WORK_DIR,SEED=$SEED"
for optional in PYTHON_BIN DEP_OVERLAY HF_CACHE; do
  if [[ -n "${!optional:-}" ]]; then
    common+=",$optional=${!optional}"
  fi
done

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=tpr-prepare \
  --time=00:20:00 --cpus-per-task=2 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" prepare)

warmup=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-warmup --time=00:45:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$prepare" \
  --output="$LOG_DIR/warmup-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" warmup)

probe=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-probe --time=01:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$warmup" \
  --output="$LOG_DIR/probe-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" probe)

smoke=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-smoke --time=00:30:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$probe" \
  --output="$LOG_DIR/smoke-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" smoke)

dense=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-dense --time=04:00:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$probe" \
  --output="$LOG_DIR/dense-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" train_dense)

routed=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-routed --time=04:00:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$smoke" \
  --output="$LOG_DIR/routed-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" train_routed)

eval_dense=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-eval-dense --time=01:15:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$dense" \
  --output="$LOG_DIR/eval-dense-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" eval_dense)

eval_routed=$(sbatch --parsable --account="$ACCOUNT" --nice="$NICE" \
  --job-name=tpr-eval-routed --time=01:15:00 --cpus-per-task=4 --mem=40G \
  --gres="$GPU" --dependency="afterok:$routed" \
  --output="$LOG_DIR/eval-routed-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" eval_routed)

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=tpr-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$eval_dense:$eval_routed" \
  --output="$LOG_DIR/collect-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_worker.sh" collect)

printf 'prepare=%s warmup=%s probe=%s smoke=%s dense=%s routed=%s eval_dense=%s eval_routed=%s collect=%s output=%s\n' \
  "$prepare" "$warmup" "$probe" "$smoke" "$dense" "$routed" \
  "$eval_dense" "$eval_routed" "$collect" "$WORK_DIR"

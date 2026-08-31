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

prepare=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-data \
  --time=00:30:00 --cpus-per-task=2 --mem=16G \
  --output="$LOG_DIR/prepare-%j.log" \
  --wrap="python3 '$SCRIPT_DIR/prepare_data.py' \
    --baseline-data-dir '$BASELINE_ROOT/data' \
    --train-source '$RELEASE_ROOT/de_novo' \
    --output-dir '$WORK_DIR/data' --replay-total 10000 \
    --seed '$SEED' --protocol '$PROTOCOL'")

train=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-train \
  --time=04:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$prepare" --output="$LOG_DIR/train-%j.log" \
  --wrap="python3 '$REPO_ROOT/ablations/joint_vs_specialists/train_arm.py' \
    --train-jsonl '$WORK_DIR/data/train.shared_property_joint.jsonl' \
    --output-dir '$WORK_DIR/model' --base-model '$MODEL' --arm joint \
    --epochs 1.0 --gradient-accumulation 32 --learning-rate 0.00008 \
    --seed '$SEED' --protocol '$PROTOCOL'")

evaluate=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-eval \
  --time=01:00:00 --cpus-per-task=4 --mem=40G --gres="$GPU" \
  --dependency="afterok:$train" --output="$LOG_DIR/eval-%j.log" \
  --wrap="python3 '$REPO_ROOT/scripts/evaluate_raw1.py' \
    --denovo-gate '$WORK_DIR/data/gate.denovo.jsonl' \
    --edit-gate '$WORK_DIR/data/gate.edit.jsonl' \
    --base-model '$MODEL' --adapter-dir '$WORK_DIR/model/adapter' \
    --arm shared_property_joint --output-dir '$WORK_DIR/eval' \
    --seed '$((SEED + 50))' --protocol '$PROTOCOL'")

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=spt-collect \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --dependency="afterok:$evaluate" --output="$LOG_DIR/collect-%j.log" \
  --wrap="python3 '$SCRIPT_DIR/collect.py' \
    --candidate-summary '$WORK_DIR/eval/summary.json' \
    --joint-summary '$BASELINE_ROOT/eval/joint/summary.json' \
    --edit-summary '$BASELINE_ROOT/eval/edit/summary.json' \
    --output-dir '$WORK_DIR/result'")

printf 'prepare=%s train=%s evaluate=%s collect=%s output=%s\n' \
  "$prepare" "$train" "$evaluate" "$collect" "$WORK_DIR"

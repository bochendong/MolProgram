#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${AUDIT_BASE_MODEL:?Set AUDIT_BASE_MODEL before submission}"
: "${AUDIT_ADAPTER_DIR:?Set AUDIT_ADAPTER_DIR before submission}"
: "${AUDIT_TRAIN_JSONL:?Set AUDIT_TRAIN_JSONL before submission}"
LABEL="${AUDIT_LABEL:-policy}"
ACCOUNT="${AUDIT_ACCOUNT:-def-hup-ab}"
GPU="${AUDIT_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUTPUT_ROOT="${AUDIT_OUTPUT_ROOT:-$REPO_ROOT/outputs/editing-support-audit/$LABEL}"
LOG_DIR="$REPO_ROOT/logs/editing-support-audit"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

job=$(sbatch --parsable --account="$ACCOUNT" --job-name="mp-audit-$LABEL" \
  --time=08:00:00 --cpus-per-task=6 --mem=40G --gres="$GPU" \
  --output="$LOG_DIR/$LABEL-%j.log" \
  --export="ALL,AUDIT_SCRIPT_DIR=$SCRIPT_DIR,AUDIT_OUTPUT_ROOT=$OUTPUT_ROOT" \
  "$SCRIPT_DIR/run.sh")

printf 'job=%s label=%s output=%s\n' "$job" "$LABEL" "$OUTPUT_ROOT"

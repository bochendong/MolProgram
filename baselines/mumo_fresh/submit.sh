#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${MUMO_TRAIN_JSONL:?Set MUMO_TRAIN_JSONL before submission}"
: "${MUMO_BASE_MODEL:?Set MUMO_BASE_MODEL before submission}"
ACCOUNT="${MUMO_ACCOUNT:-def-hup-ab}"
GPU="${MUMO_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT_ROOT="${MUMO_OUTPUT_ROOT:-$REPO_ROOT/outputs/baselines/mumo_fresh/seed_32002}"
LOG_DIR="$REPO_ROOT/logs/baselines/mumo_fresh"
mkdir -p "$LOG_DIR"
common="MUMO_BASELINE_DIR=$SCRIPT_DIR,MUMO_OUTPUT_ROOT=$OUT_ROOT"

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-check \
  --time=00:10:00 --cpus-per-task=2 --mem=8G \
  --output="$LOG_DIR/preflight-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_preflight.sh")
smoke=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-smoke \
  --time=01:00:00 --cpus-per-task=6 --mem=40G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/smoke-%j.log" \
  --export="ALL,$common,MUMO_RUN_KIND=smoke" "$SCRIPT_DIR/run_train.sh")
full=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-full \
  --time=12:00:00 --cpus-per-task=6 --mem=40G --gres="$GPU" \
  --dependency="afterok:$smoke" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/full-%j.log" \
  --export="ALL,$common,MUMO_RUN_KIND=full" "$SCRIPT_DIR/run_train.sh")
validate=$(sbatch --parsable --account="$ACCOUNT" --job-name=mumo-validate \
  --time=00:30:00 --cpus-per-task=2 --mem=16G \
  --dependency="afterok:$full" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/validate-%j.log" --export="ALL,$common" \
  "$SCRIPT_DIR/run_validate.sh")

printf 'preflight=%s smoke=%s full=%s validate=%s output=%s\n' \
  "$preflight" "$smoke" "$full" "$validate" "$OUT_ROOT"

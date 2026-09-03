#!/usr/bin/env bash
set -euo pipefail

# Required environment:
# SAFE_GRPO_BASE_MODEL, SAFE_GRPO_INPUT_ADAPTER, SAFE_GRPO_INPUT_MARKER,
# SAFE_GRPO_TRAIN_JSONL, SAFE_GRPO_DENOVO_DEV, SAFE_GRPO_EDIT_DEV,
# SAFE_GRPO_DENOVO_FINAL, SAFE_GRPO_EDIT_FINAL, SAFE_GRPO_ASSAY_ORACLE_DIR.
# Optional SAFE_GRPO_DEPENDENCY is the upstream fresh-balanced Slurm job id.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
for name in SAFE_GRPO_BASE_MODEL SAFE_GRPO_INPUT_ADAPTER SAFE_GRPO_INPUT_MARKER \
  SAFE_GRPO_TRAIN_JSONL SAFE_GRPO_DENOVO_DEV SAFE_GRPO_EDIT_DEV \
  SAFE_GRPO_DENOVO_FINAL SAFE_GRPO_EDIT_FINAL SAFE_GRPO_ASSAY_ORACLE_DIR; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done

ACCOUNT="${SAFE_GRPO_ACCOUNT:-def-hup-ab}"
GPU="${SAFE_GRPO_GRES:-gpu:nvidia_h100_80gb_hbm3_3g.40gb:1}"
OUT="${SAFE_GRPO_OUTPUT_ROOT:-$ROOT/outputs/safe-joint-raw1-grpo/seed-37001}"
LOG_DIR="${SAFE_GRPO_LOG_DIR:-$ROOT/logs/safe-joint-raw1-grpo}"
SUPPORT="$OUT/support/support_report.json"
mkdir -p "$OUT" "$LOG_DIR"

dependency_args=()
[[ -n "${SAFE_GRPO_DEPENDENCY:-}" ]] && dependency_args+=(
  --dependency="afterok:$SAFE_GRPO_DEPENDENCY" --kill-on-invalid-dep=yes
)
common_export="ALL,SAFE_GRPO_REPO_ROOT=$ROOT,SAFE_GRPO_SCRIPT_DIR=$SCRIPT_DIR,SAFE_GRPO_BASE_MODEL=$SAFE_GRPO_BASE_MODEL,SAFE_GRPO_INPUT_ADAPTER=$SAFE_GRPO_INPUT_ADAPTER,SAFE_GRPO_INPUT_MARKER=$SAFE_GRPO_INPUT_MARKER,SAFE_GRPO_TRAIN_JSONL=$SAFE_GRPO_TRAIN_JSONL,SAFE_GRPO_DENOVO_DEV=$SAFE_GRPO_DENOVO_DEV,SAFE_GRPO_EDIT_DEV=$SAFE_GRPO_EDIT_DEV,SAFE_GRPO_DENOVO_FINAL=$SAFE_GRPO_DENOVO_FINAL,SAFE_GRPO_EDIT_FINAL=$SAFE_GRPO_EDIT_FINAL,SAFE_GRPO_ASSAY_ORACLE_DIR=$SAFE_GRPO_ASSAY_ORACLE_DIR,SUCC_GSK3B_ORACLE_PATH=$SAFE_GRPO_ASSAY_ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl,SUCC_DRD2_ORACLE_PATH=$SAFE_GRPO_ASSAY_ORACLE_DIR/drd2_graph2graph_svc_py36.pkl,SAFE_GRPO_OUTPUT_ROOT=$OUT,SAFE_GRPO_SUPPORT_REPORT=$SUPPORT"

preflight=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-check \
  --time=00:15:00 --cpus-per-task=2 --mem=8G "${dependency_args[@]}" \
  --output="$LOG_DIR/preflight-%j.log" --export="$common_export" \
  "$SCRIPT_DIR/run_preflight.sh")

audit=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-support \
  --time=08:00:00 --cpus-per-task=6 --mem=40G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/support-%j.log" \
  --export="$common_export,AUDIT_SCRIPT_DIR=$ROOT/audits/editing_reward_support,AUDIT_BASE_MODEL=$SAFE_GRPO_BASE_MODEL,AUDIT_ADAPTER_DIR=$SAFE_GRPO_INPUT_ADAPTER,AUDIT_TRAIN_JSONL=$SAFE_GRPO_TRAIN_JSONL,AUDIT_OUTPUT_ROOT=$OUT/support,AUDIT_PYTHON_BIN=${SAFE_GRPO_PYTHON_BIN:-python},AUDIT_DEP_OVERLAY=${SAFE_GRPO_DEP_OVERLAY:-$ROOT/src}" \
  "$ROOT/audits/editing_reward_support/run.sh")

rl=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-joint-rl \
  --time=03:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$audit" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/rl-%j.log" --export="$common_export,SAFE_GRPO_ARM=rl" \
  "$SCRIPT_DIR/run_arm.sh")

control=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-sft-ctrl \
  --time=01:30:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/control-%j.log" \
  --export="$common_export,SAFE_GRPO_ARM=continued_sft" "$SCRIPT_DIR/run_arm.sh")

dev_eval=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-dev-raw1 \
  --time=04:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$rl:$control" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/dev-eval-%j.log" --export="$common_export" \
  "$SCRIPT_DIR/run_dev_eval.sh")

select=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-select \
  --time=00:10:00 --cpus-per-task=2 --mem=4G \
  --dependency="afterok:$dev_eval" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/select-%j.log" --export="$common_export" \
  "$SCRIPT_DIR/run_select.sh")

final_eval=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-final-raw1 \
  --time=02:00:00 --cpus-per-task=4 --mem=48G --gres="$GPU" \
  --dependency="afterok:$select" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/final-eval-%j.log" --export="$common_export" \
  "$SCRIPT_DIR/run_final_eval.sh")

collect=$(sbatch --parsable --account="$ACCOUNT" --job-name=safegrpo-collect \
  --time=00:15:00 --cpus-per-task=2 --mem=4G \
  --dependency="afterok:$final_eval" --kill-on-invalid-dep=yes \
  --output="$LOG_DIR/collect-%j.log" --export="$common_export" \
  "$SCRIPT_DIR/run_collect.sh")

printf 'preflight_job=%s\nsupport_job=%s\nrl_job=%s\ncontrol_job=%s\n' \
  "$preflight" "$audit" "$rl" "$control"
printf 'dev_eval_job=%s\nselection_job=%s\nfinal_eval_job=%s\ncollect_job=%s\noutput=%s\n' \
  "$dev_eval" "$select" "$final_eval" "$collect" "$OUT"

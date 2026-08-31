#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${MUMO_PYTHON_BIN:-python}"
OUT_ROOT="${MUMO_OUTPUT_ROOT:-$REPO_ROOT/outputs/baselines/mumo_fresh/seed_32002}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11
"$PY" "$SCRIPT_DIR/validate_adapter.py" \
  --adapter-dir "$OUT_ROOT/full/adapter" \
  --output-json "$OUT_ROOT/full/finite_audit.json"
touch "$OUT_ROOT/FULL_VALIDATED"

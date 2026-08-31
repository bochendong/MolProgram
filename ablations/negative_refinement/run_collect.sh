#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${NEGATIVE_ABLATION_DIR:?NEGATIVE_ABLATION_DIR must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_ROOT="${NEGATIVE_OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/negative_refinement/seed_2323}"
PY="${NEGATIVE_PYTHON_BIN:-python}"
module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11
"$PY" "$SCRIPT_DIR/collect.py" --output-root "$OUT_ROOT"

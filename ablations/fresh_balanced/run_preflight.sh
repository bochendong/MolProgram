#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${FRESH_SCRIPT_DIR:?FRESH_SCRIPT_DIR must be exported}"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
: "${FRESH_MODEL:?FRESH_MODEL is required}"
: "${FRESH_DATA_ROOT:?FRESH_DATA_ROOT is required}"
PY="${FRESH_PYTHON_BIN:-python}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4
export PYTHONPATH="${FRESH_DEP_OVERLAY:-$ROOT/src}:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

test -f "$FRESH_MODEL/config.json"
test -f "$FRESH_DATA_ROOT/RELEASE_COMPLETE"
"$PY" -m py_compile "$ROOT/scripts/train_indexed_sft.py"
"$PY" -m pytest -q "$ROOT/tests/test_fresh_balanced_contract.py"

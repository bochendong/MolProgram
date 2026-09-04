#!/usr/bin/env python3
"""Freeze the edit-only training subset for the conditional residual pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
from molprogram import program_routing  # noqa: E402


EDIT_ONLY_PROPERTIES = frozenset({"SA", "GSK3B", "DRD2"})
EDIT_ONLY_TASKS = frozenset(
    {
        "DRD2:decrease+MW:decrease+SA:decrease",
        "GSK3B:increase",
        "HBA:decrease+SA:decrease",
        "QED:increase+SA:decrease",
        "SA:decrease",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_edit_only(row: Mapping[str, object]) -> bool:
    return (
        program_routing.task_mode(row) == "edit"
        and bool(set(program_routing.properties(row)) & EDIT_ONLY_PROPERTIES)
    )


def select_rows(
    rows: Sequence[dict[str, object]], *, expected_rows: int
) -> tuple[list[dict[str, object]], Counter[str]]:
    selected = [row for row in rows if is_edit_only(row)]
    tasks = Counter(str(row.get("task_key", "")) for row in selected)
    if len(selected) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} edit-only rows, found {len(selected)}"
        )
    if set(tasks) != EDIT_ONLY_TASKS:
        raise ValueError(
            f"expected edit-only tasks {sorted(EDIT_ONLY_TASKS)}, found {sorted(tasks)}"
        )
    if len(set(tasks.values())) != 1:
        raise ValueError(f"edit-only task counts are not balanced: {tasks}")
    return selected, tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=1920)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.source_train.read_text().splitlines()
        if line.strip()
    ]
    selected, tasks = select_rows(rows, expected_rows=args.expected_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "train.edit_only.jsonl"
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    )
    manifest = {
        "protocol": "property_conditional_residual_rank4_pilot_v1",
        "source_train": str(args.source_train.resolve()),
        "source_sha256": sha256(args.source_train),
        "output_sha256": sha256(output),
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "task_counts": dict(sorted(tasks.items())),
        "edit_only_properties": sorted(EDIT_ONLY_PROPERTIES),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

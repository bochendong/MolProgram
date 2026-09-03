#!/usr/bin/env python3
"""Create disjoint, bucket-balanced dev/final gates from mode-specific gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from molprogram.safe_grpo import DE_NOVO_BUCKETS, EDIT_BUCKETS, balanced_bucket


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: Mapping[str, object]) -> str:
    value = row.get("condition_id", row.get("example_id", row.get("sample_id", "")))
    if not value:
        raise ValueError("row is missing condition_id/example_id/sample_id")
    return str(value)


def stable_key(row: Mapping[str, object], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identity(row)}".encode()).hexdigest()


def stratified_split(
    rows: list[dict[str, object]],
    buckets: tuple[str, ...],
    per_split: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[balanced_bucket(row)].append(row)
    dev: list[dict[str, object]] = []
    final: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for bucket in buckets:
        candidates = sorted(grouped[bucket], key=lambda row: stable_key(row, seed))
        required = 2 * per_split
        if len(candidates) < required:
            raise ValueError(
                f"{bucket} has {len(candidates)} rows; {required} are required"
            )
        left = candidates[:per_split]
        right = candidates[per_split:required]
        for row in left + right:
            row_id = identity(row)
            if row_id in used_ids:
                raise ValueError(f"duplicate identity in source gates: {row_id}")
            used_ids.add(row_id)
        dev.extend(left)
        final.extend(right)
    return dev, final, len(rows) - len(dev) - len(final)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--denovo-jsonl", required=True, type=Path)
    parser.add_argument("--edit-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--denovo-per-split", type=int, default=10)
    parser.add_argument("--edit-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=37001)
    args = parser.parse_args()

    denovo_rows = read_jsonl(args.denovo_jsonl)
    edit_rows = read_jsonl(args.edit_jsonl)
    dev_denovo, final_denovo, unused_denovo = stratified_split(
        denovo_rows, DE_NOVO_BUCKETS, args.denovo_per_split, args.seed
    )
    dev_edit, final_edit, unused_edit = stratified_split(
        edit_rows, EDIT_BUCKETS, args.edit_per_split, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dev_denovo": args.output_dir / "dev.denovo.jsonl",
        "dev_edit": args.output_dir / "dev.edit.jsonl",
        "final_denovo": args.output_dir / "final.denovo.jsonl",
        "final_edit": args.output_dir / "final.edit.jsonl",
    }
    split_rows = {
        "dev_denovo": dev_denovo,
        "dev_edit": dev_edit,
        "final_denovo": final_denovo,
        "final_edit": final_edit,
    }
    for name, path in outputs.items():
        write_jsonl(path, split_rows[name])

    manifest = {
        "protocol": "molprogram_safe_joint_raw1_stratified_frozen_inputs_v1",
        "seed": args.seed,
        "train_jsonl": str(args.train_jsonl.resolve()),
        "source_gates": {
            "denovo": str(args.denovo_jsonl.resolve()),
            "edit": str(args.edit_jsonl.resolve()),
        },
        "source_sha256": {
            "train": sha256(args.train_jsonl),
            "denovo": sha256(args.denovo_jsonl),
            "edit": sha256(args.edit_jsonl),
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "row_counts": {name: len(rows) for name, rows in split_rows.items()},
        "per_bucket_per_split": {
            "de_novo": args.denovo_per_split,
            "edit": args.edit_per_split,
        },
        "unused_source_rows": {"de_novo": unused_denovo, "edit": unused_edit},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "INPUTS_FROZEN").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

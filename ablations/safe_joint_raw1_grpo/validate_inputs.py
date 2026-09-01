#!/usr/bin/env python3
"""Validate frozen train/dev/final inputs before allocating experiment GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from molprogram import protocol
from molprogram.safe_grpo import DE_NOVO_BUCKETS, EDIT_BUCKETS, balanced_bucket


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: Mapping[str, object]) -> str:
    value = row.get("condition_id", row.get("example_id", row.get("sample_id", "")))
    if not value:
        raise ValueError("row is missing condition_id/example_id/sample_id")
    return str(value)


def validate_messages(rows: list[dict[str, object]], *, require_answer: bool) -> None:
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"row {identity(row)} has no structured messages")
        if require_answer and str(messages[-1].get("role", "")) != "assistant":
            raise ValueError(f"training row {identity(row)} has no SFT answer")


def require_buckets(
    rows: list[dict[str, object]],
    expected: tuple[str, ...],
    label: str,
    *,
    minimum: int = 1,
) -> dict[str, int]:
    counts = Counter(balanced_bucket(row) for row in rows)
    missing = [name for name in expected if counts[name] < minimum]
    if missing:
        raise ValueError(f"{label} has buckets below minimum {minimum}: {missing}")
    return {name: counts[name] for name in expected}


def assert_disjoint(named_ids: dict[str, set[str]]) -> None:
    labels = list(named_ids)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            overlap = named_ids[left] & named_ids[right]
            if overlap:
                raise ValueError(
                    f"{left} and {right} overlap on frozen identities: {sorted(overlap)[:5]}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--denovo-dev", required=True, type=Path)
    parser.add_argument("--edit-dev", required=True, type=Path)
    parser.add_argument("--denovo-final", required=True, type=Path)
    parser.add_argument("--edit-final", required=True, type=Path)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--input-marker", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "train": args.train_jsonl,
        "denovo_dev": args.denovo_dev,
        "edit_dev": args.edit_dev,
        "denovo_final": args.denovo_final,
        "edit_final": args.edit_final,
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    validate_messages(rows["train"], require_answer=True)
    for name in ("denovo_dev", "edit_dev", "denovo_final", "edit_final"):
        validate_messages(rows[name], require_answer=False)

    train_de = [row for row in rows["train"] if row.get("task_mode") == "de_novo"]
    train_edit = [row for row in rows["train"] if row.get("task_mode") == "edit"]
    bucket_counts = {
        "train_de_novo": require_buckets(
            train_de, DE_NOVO_BUCKETS, "training de novo", minimum=5
        ),
        "train_edit": require_buckets(
            train_edit, EDIT_BUCKETS, "training edit", minimum=3
        ),
        "denovo_dev": require_buckets(rows["denovo_dev"], DE_NOVO_BUCKETS, "de novo dev"),
        "edit_dev": require_buckets(rows["edit_dev"], EDIT_BUCKETS, "edit dev"),
        "denovo_final": require_buckets(rows["denovo_final"], DE_NOVO_BUCKETS, "de novo final"),
        "edit_final": require_buckets(rows["edit_final"], EDIT_BUCKETS, "edit final"),
    }
    for name in ("denovo_dev", "denovo_final"):
        if any(row.get("task_mode") != "de_novo" for row in rows[name]):
            raise ValueError(f"{name} contains non-de-novo rows")
    for name in ("edit_dev", "edit_final"):
        if any(row.get("task_mode") != "edit" for row in rows[name]):
            raise ValueError(f"{name} contains non-editing rows")

    split_ids = {
        "train": {identity(row) for row in rows["train"]},
        "dev": {
            identity(row) for name in ("denovo_dev", "edit_dev") for row in rows[name]
        },
        "final": {
            identity(row)
            for name in ("denovo_final", "edit_final")
            for row in rows[name]
        },
    }
    expected_dev_rows = len(rows["denovo_dev"]) + len(rows["edit_dev"])
    expected_final_rows = len(rows["denovo_final"]) + len(rows["edit_final"])
    if len(split_ids["dev"]) != expected_dev_rows:
        raise ValueError("development gate contains duplicate frozen identities")
    if len(split_ids["final"]) != expected_final_rows:
        raise ValueError("final gate contains duplicate frozen identities")
    assert_disjoint(split_ids)
    adapter_file = args.input_adapter / "adapter_model.safetensors"
    if not adapter_file.is_file() or not args.input_marker.is_file():
        raise FileNotFoundError("fresh input adapter or completion marker is missing")

    result = {
        "protocol": "molprogram_safe_joint_raw1_input_validation_v1",
        "passed": True,
        "registered_edit_tasks": sorted(protocol.TABLE1_TASK_KEYS.values()),
        "row_counts": {name: len(values) for name, values in rows.items()},
        "bucket_counts": bucket_counts,
        "split_identity_counts": {name: len(values) for name, values in split_ids.items()},
        "sha256": {
            **{name: sha256(path) for name, path in paths.items()},
            "input_adapter": sha256(adapter_file),
            "input_marker": sha256(args.input_marker),
        },
        "input_adapter": str(args.input_adapter.resolve()),
        "input_marker": str(args.input_marker.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

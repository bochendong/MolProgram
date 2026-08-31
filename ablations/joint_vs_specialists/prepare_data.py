#!/usr/bin/env python3
"""Freeze matched joint/specialist subsets and target-blind Raw@1 gates."""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


EDIT_TASKS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "HBA:decrease+SA:decrease",
    "HBA:increase+MW:increase+QED:decrease",
    "MW:increase",
    "QED:increase+SA:decrease",
    "RB:decrease",
    "SA:decrease",
)


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    return rows


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("condition_id", row.get("sample_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def property_count(row: Mapping[str, object]) -> int:
    program = row.get("condition_program")
    if isinstance(program, list):
        return len(program)
    for message in list(row.get("messages", [])):
        if str(message.get("role")) == "user":
            payload = json.loads(str(message.get("content", "{}")))
            conditions = payload.get("conditions", []) if isinstance(payload, dict) else []
            return len(conditions) if isinstance(conditions, list) else 0
    return 0


def balanced_quotas(keys: Sequence[str], total: int) -> dict[str, int]:
    if total < len(keys):
        raise ValueError(f"total={total} is smaller than {len(keys)} buckets")
    base, remainder = divmod(total, len(keys))
    return {key: base + (index < remainder) for index, key in enumerate(keys)}


def training_bucket(row: Mapping[str, object], edit_grouping: str) -> str | None:
    mode = str(row.get("task_mode", ""))
    count = property_count(row)
    if mode == "de_novo" and 2 <= count <= 7:
        return f"de_novo:{count}p"
    if mode == "edit":
        if edit_grouping == "property_count" and 1 <= count <= 7:
            return f"edit:{count}p"
        task = str(row.get("task_key", ""))
        if edit_grouping == "benchmark_task" and task in EDIT_TASKS:
            return f"edit:{task}"
    return None


def training_keys(edit_grouping: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    de_keys = tuple(f"de_novo:{count}p" for count in range(2, 8))
    if edit_grouping == "property_count":
        edit_keys = tuple(f"edit:{count}p" for count in range(1, 8))
    elif edit_grouping == "benchmark_task":
        edit_keys = tuple(f"edit:{task}" for task in EDIT_TASKS)
    else:
        raise ValueError(f"unsupported edit grouping: {edit_grouping}")
    return de_keys, edit_keys


def assemble_training(
    selected: Mapping[str, list[dict[str, object]]],
    de_keys: Sequence[str],
    edit_keys: Sequence[str],
):
    de_novo = [row for key in de_keys for row in selected[key]]
    editing = [row for key in edit_keys for row in selected[key]]
    joint: list[dict[str, object]] = []
    for index in range(max(len(de_novo), len(editing))):
        if index < len(de_novo):
            joint.append(de_novo[index])
        if index < len(editing):
            joint.append(editing[index])
    return de_novo, editing, joint


def select_train(
    rows: Sequence[dict[str, object]],
    seed: int,
    *,
    denovo_total: int = 3000,
    edit_total: int = 3000,
    edit_grouping: str = "benchmark_task",
):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = training_bucket(row, edit_grouping)
        if bucket:
            grouped[bucket].append(row)

    de_keys, edit_keys = training_keys(edit_grouping)
    quotas = {
        **balanced_quotas(de_keys, denovo_total),
        **balanced_quotas(edit_keys, edit_total),
    }
    selected: dict[str, list[dict[str, object]]] = {}
    for key, quota in quotas.items():
        values = sorted(grouped[key], key=lambda row: stable_key(row, seed))
        if len(values) < quota:
            raise ValueError(f"bucket {key} has {len(values)} rows; needs {quota}")
        selected[key] = values[:quota]

    de_novo, editing, joint = assemble_training(selected, de_keys, edit_keys)
    return de_novo, editing, joint, quotas


def select_train_streaming(
    paths: Sequence[Path],
    seed: int,
    *,
    denovo_total: int,
    edit_total: int,
    edit_grouping: str,
):
    """Select deterministic smallest-hash rows without loading the 2.57M corpus."""
    de_keys, edit_keys = training_keys(edit_grouping)
    quotas = {
        **balanced_quotas(de_keys, denovo_total),
        **balanced_quotas(edit_keys, edit_total),
    }
    heaps: dict[str, list[tuple[int, int, dict[str, object]]]] = {
        key: [] for key in quotas
    }
    counter = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket = training_bucket(row, edit_grouping)
                if bucket not in quotas:
                    continue
                counter += 1
                item = (-int(stable_key(row, seed), 16), counter, row)
                heap = heaps[bucket]
                if len(heap) < quotas[bucket]:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    selected: dict[str, list[dict[str, object]]] = {}
    for key, quota in quotas.items():
        if len(heaps[key]) < quota:
            raise ValueError(f"bucket {key} has {len(heaps[key])} rows; needs {quota}")
        selected[key] = [item[2] for item in sorted(heaps[key], reverse=True)]
    de_novo, editing, joint = assemble_training(selected, de_keys, edit_keys)
    return de_novo, editing, joint, quotas


def select_denovo_gate(
    rows: Sequence[dict[str, object]], seed: int, *, per_bucket: int = 20
):
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identity = str(row.get("condition_id", row.get("sample_id", "")))
        count = property_count(row)
        if identity and identity not in seen and 2 <= count <= 7:
            grouped[count].append(row)
            seen.add(identity)
    selected: list[dict[str, object]] = []
    for count in range(2, 8):
        values = sorted(grouped[count], key=lambda row: stable_key(row, seed))
        if len(values) < per_bucket:
            raise ValueError(f"gate has only {len(values)} rows for {count}p")
        selected.extend(values[:per_bucket])
    return selected


def select_edit_gate(
    rows: Sequence[dict[str, object]], seed: int, *, per_bucket: int = 20
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identity = str(row.get("condition_id", row.get("sample_id", "")))
        task = str(row.get("task_key", ""))
        if (
            str(row.get("task_mode", "")) == "edit"
            and task in EDIT_TASKS
            and identity
            and identity not in seen
        ):
            grouped[task].append(row)
            seen.add(identity)
    selected: list[dict[str, object]] = []
    for task in EDIT_TASKS:
        values = sorted(grouped[task], key=lambda row: stable_key(row, seed))
        if len(values) < per_bucket:
            raise ValueError(
                f"edit gate has only {len(values)} rows for {task}; needs {per_bucket}"
            )
        selected.extend(values[:per_bucket])
    return selected


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", action="append", required=True, type=Path)
    parser.add_argument("--denovo-prompts", action="append", required=True, type=Path)
    parser.add_argument("--edit-prompts", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=33001)
    parser.add_argument("--denovo-train-total", type=int, default=3000)
    parser.add_argument("--edit-train-total", type=int, default=3000)
    parser.add_argument("--gate-per-bucket", type=int, default=20)
    parser.add_argument("--denovo-gate-per-bucket", type=int)
    parser.add_argument("--edit-gate-per-bucket", type=int)
    parser.add_argument(
        "--edit-grouping",
        choices=("benchmark_task", "property_count"),
        default="benchmark_task",
    )
    parser.add_argument(
        "--protocol", default="molprogram_joint_vs_specialist_v1"
    )
    args = parser.parse_args(argv)
    denovo_gate_per_bucket = args.denovo_gate_per_bucket or args.gate_per_bucket
    edit_gate_per_bucket = args.edit_gate_per_bucket or args.gate_per_bucket

    train_paths: list[Path] = []
    for source in args.train_source:
        if source.is_dir():
            train_paths.extend(sorted(source.glob("*.jsonl")))
        else:
            train_paths.append(source)
    if not train_paths:
        raise ValueError("no training JSONL shards found")
    de_novo, editing, joint, quotas = select_train_streaming(
        train_paths,
        args.seed,
        denovo_total=args.denovo_train_total,
        edit_total=args.edit_train_total,
        edit_grouping=args.edit_grouping,
    )
    gate = select_denovo_gate(
        read_jsonl(args.denovo_prompts),
        args.seed + 1,
        per_bucket=denovo_gate_per_bucket,
    )
    write_jsonl(args.output_dir / "train.denovo.jsonl", de_novo)
    write_jsonl(args.output_dir / "train.edit.jsonl", editing)
    write_jsonl(args.output_dir / "train.joint.jsonl", joint)
    write_jsonl(args.output_dir / "gate.denovo.jsonl", gate)
    edit_gate: list[dict[str, object]] = []
    if args.edit_prompts:
        edit_gate = select_edit_gate(
            read_jsonl([args.edit_prompts]),
            args.seed + 2,
            per_bucket=edit_gate_per_bucket,
        )
        write_jsonl(args.output_dir / "gate.edit.jsonl", edit_gate)
    summary = {
        "protocol": args.protocol,
        "sources": [str(path) for path in train_paths],
        "seed": args.seed,
        "rows": {"joint": len(joint), "denovo": len(de_novo), "edit": len(editing)},
        "bucket_quotas": quotas,
        "denovo_gate_rows": len(gate),
        "edit_gate_rows": len(edit_gate),
        "denovo_gate_per_bucket": denovo_gate_per_bucket,
        "edit_gate_per_bucket": edit_gate_per_bucket,
        "edit_grouping": args.edit_grouping,
        "matched_examples": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

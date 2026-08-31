#!/usr/bin/env python3
"""Build the audited MolProgramInstruct release and indexed training shards."""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SOURCE_DIR))
from molprogram import protocol  # noqa: E402

DESCRIPTOR_PROPERTIES = ("MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB")


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def rank64(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def read_jsonl(paths: list[Path]) -> Iterable[dict[str, object]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def heldout_hashes(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if path.suffix == ".jsonl":
            rows: Iterable[Mapping[str, object]] = read_jsonl([path])
        else:
            handle = path.open(newline="", encoding="utf-8")
            rows = csv.DictReader(handle)
        try:
            for row in rows:
                for key in (
                    "source_canonical_smiles", "source_smiles",
                    "target_canonical_smiles", "target_smiles",
                ):
                    smiles = str(row.get(key, "") or "").strip()
                    canonical = protocol.canonical_smiles(smiles) if smiles else ""
                    if canonical:
                        result.add(hashlib.sha256(canonical.encode()).hexdigest())
        finally:
            if path.suffix != ".jsonl":
                handle.close()
    return result


def select_cutoff(ranks: array.array, target: int) -> int:
    if len(ranks) < target:
        raise ValueError(f"only {len(ranks)} eligible unique rows; requested {target}")
    values = np.frombuffer(ranks, dtype=np.uint64)
    return int(np.partition(values, target - 1)[target - 1])


def ordered_subset(items: list[str], size: int, material: str) -> list[str]:
    return sorted(
        sorted(items, key=lambda item: hashlib.sha256(f"{material}:{item}".encode()).hexdigest())[:size],
        key=lambda item: protocol.PROPERTY_ORDER[item],
    )


def balanced_quotas(total: int, buckets: list[int]) -> dict[int, int]:
    base, remainder = divmod(total, len(buckets))
    return {bucket: base + int(index < remainder) for index, bucket in enumerate(buckets)}


def assigned_bucket(
    feasible: list[int], counts: Counter[int], material: str,
) -> int:
    """Choose the least-populated feasible task bucket with deterministic ties."""
    return min(
        feasible,
        key=lambda bucket: (
            counts[bucket], hashlib.sha256(f"{material}:bucket:{bucket}".encode()).hexdigest(),
        ),
    )


def messages(source: str, program: list[dict[str, object]], target: str, mode: str):
    user = json.dumps(
        {"conditions": program, "source": source}, sort_keys=True, separators=(",", ":")
    )
    assistant = protocol.response(target, mode)
    return [
        {"role": "system", "content": protocol.SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


class ShardWriter:
    def __init__(self, output_dir: Path, mode: str, shards: int):
        self.root = output_dir / mode
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = [self.root / f"train-{i:05d}-of-{shards:05d}.jsonl" for i in range(shards)]
        self.index_paths = [path.with_suffix(".idx") for path in self.paths]
        self.handles = [path.open("wb") for path in self.paths]
        self.indices = [path.open("wb") for path in self.index_paths]
        self.counts = [0] * shards
        self.total = 0

    def write(self, row: Mapping[str, object]) -> None:
        shard = self.total % len(self.handles)
        payload = (json.dumps(dict(row), sort_keys=True) + "\n").encode()
        offset = self.handles[shard].tell()
        self.indices[shard].write(struct.pack("<Q", offset))
        self.handles[shard].write(payload)
        self.counts[shard] += 1
        self.total += 1

    def close(self) -> dict[str, object]:
        for handle in [*self.handles, *self.indices]:
            handle.close()
        return {
            "rows": self.total,
            "shards": len(self.paths),
            "rows_per_shard": self.counts,
            "files": [
                {
                    "jsonl": str(path),
                    "index": str(index),
                    "rows": count,
                    "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                }
                for path, index, count in zip(self.paths, self.index_paths, self.counts)
            ],
        }


def de_novo_record(
    row: Mapping[str, object], seed: int, property_count: int | None = None,
) -> dict[str, object]:
    target = str(row["target_smiles"])
    rank = str(row["selection_rank"])
    values = dict(row["descriptor_values"])
    count = property_count if property_count is not None else 2 + rank64(f"{seed}:count:{rank}") % 6
    properties = ordered_subset(list(DESCRIPTOR_PROPERTIES), count, f"{seed}:{rank}")
    program = [
        {"property": prop, "goal": {"around": values[prop]}}
        for prop in properties
    ]
    target_hash = str(row["target_hash"])
    return {
        "dataset": "MolProgramInstruct-Balanced",
        "release_version": "1.0",
        "example_id": f"mpi4m:de_novo:{target_hash[:24]}",
        "task_mode": "de_novo",
        "source_smiles": "<EMPTY>",
        "target_smiles": target,
        "source_hash": "",
        "target_hash": target_hash,
        "condition_program": program,
        "condition_hash": protocol.condition_hash_from_program(program),
        "task_key": protocol.task_key(program),
        "property_count": count,
        "messages": messages("<EMPTY>", program, target, "de_novo"),
        "provenance": {
            "source_dataset": "PubChem",
            "source_record_id": row.get("source_record_id", ""),
            "descriptor_values": values,
            "program_view": "deterministic_property_subset",
        },
        "selection_rank": rank,
    }


def edit_fields(row: Mapping[str, object]) -> tuple[str, str, list[str]] | None:
    if str(row.get("source_valid", "")).strip() and not truthy(row.get("source_valid")):
        return None
    if str(row.get("target_valid", "")).strip() and not truthy(row.get("target_valid")):
        return None
    source = str(row.get("source_canonical_smiles", "") or row.get("source_smiles", "")).strip()
    target = str(row.get("target_canonical_smiles", "") or row.get("target_smiles", "")).strip()
    if not source or not target or source == target:
        return None
    try:
        if float(row.get("source_target_tanimoto", 0.0) or 0.0) < 0.15:
            return None
    except ValueError:
        return None
    active = [prop for prop in DESCRIPTOR_PROPERTIES if truthy(row.get(f"{prop}_active"))]
    return (source, target, active) if active else None


def edit_record(
    row: Mapping[str, object], source: str, target: str, active: list[str], seed: int,
    property_count: int | None = None,
):
    raw_id = str(row.get("example_id", "") or row.get("sample_id", ""))
    material = f"{seed}:{raw_id}:{source}:{target}"
    rank = hashlib.sha256(material.encode()).hexdigest()
    count = property_count if property_count is not None else 1 + rank64(f"{seed}:count:{material}") % min(7, len(active))
    properties = ordered_subset(active, count, material)
    program = [
        {"property": prop, "goal": protocol.canonical_direction(row.get(f"{prop}_direction", ""))}
        for prop in properties
    ]
    if any(not item["goal"] for item in program):
        raise ValueError(f"active edit property lacks direction: {raw_id}")
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    target_hash = hashlib.sha256(target.encode()).hexdigest()
    pair_hash = hashlib.sha256(f"{source}\n{target}".encode()).hexdigest()
    return {
        "dataset": "MolProgramInstruct-Balanced",
        "release_version": "1.0",
        "example_id": f"mpi4m:edit:{pair_hash[:24]}",
        "task_mode": "edit",
        "source_smiles": source,
        "target_smiles": target,
        "source_hash": source_hash,
        "target_hash": target_hash,
        "pair_hash": pair_hash,
        "source_tanimoto": float(row.get("source_target_tanimoto", 0.0)),
        "condition_program": program,
        "condition_hash": protocol.condition_hash_from_program(program),
        "task_key": protocol.task_key(program),
        "property_count": count,
        "messages": messages(source, program, target, "edit"),
        "provenance": {
            "source_dataset": "MolEdit-Instruct",
            "source_record_id": raw_id,
            "original_instruction": row.get("instruction", ""),
            "original_instruction_tasks": row.get("instruction_tasks", ""),
            "verified_descriptor_properties": active,
            "program_view": "target_delta_verified_property_subset",
        },
        "selection_rank": rank,
    }


def build_denovo(args: argparse.Namespace) -> dict[str, object]:
    frozen = heldout_hashes(args.heldout)
    buckets = list(range(2, 8))
    quotas = balanced_quotas(args.target_rows, buckets)
    ranks = {bucket: array.array("Q") for bucket in buckets}
    seen: set[int] = set()
    audit: Counter[str] = Counter()
    for row in read_jsonl(args.input):
        audit["input"] += 1
        target_hash = str(row.get("target_hash", ""))
        if target_hash in frozen:
            audit["heldout_overlap"] += 1
            continue
        unique = int(target_hash[:16], 16)
        if unique in seen:
            audit["duplicate_target"] += 1
            continue
        seen.add(unique)
        bucket = 2 + rank64(f"{args.seed}:task:{target_hash}") % 6
        ranks[bucket].append(int(str(row["selection_rank"])[:16], 16))
        audit["eligible_unique"] += 1
    cutoffs = {bucket: select_cutoff(ranks[bucket], quotas[bucket]) for bucket in buckets}
    writer = ShardWriter(args.output_dir, "de_novo", args.shards)
    seen.clear()
    distribution: Counter[str] = Counter()
    for row in read_jsonl(args.input):
        target_hash = str(row.get("target_hash", ""))
        if target_hash in frozen:
            continue
        unique = int(target_hash[:16], 16)
        if unique in seen:
            continue
        seen.add(unique)
        bucket = 2 + rank64(f"{args.seed}:task:{target_hash}") % 6
        if int(str(row["selection_rank"])[:16], 16) > cutoffs[bucket]:
            continue
        record = de_novo_record(row, args.seed, bucket)
        writer.write(record)
        distribution[f"{record['property_count']}p"] += 1
        if writer.total == args.target_rows:
            break
    files = writer.close()
    if writer.total != args.target_rows:
        raise AssertionError(f"wrote {writer.total} de novo rows, expected {args.target_rows}")
    expected_distribution = {f"{key}p": value for key, value in quotas.items()}
    if dict(sorted(distribution.items())) != expected_distribution:
        raise AssertionError(
            f"de novo task distribution is not exactly balanced: {dict(distribution)}"
        )
    return {
        "mode": "de_novo", "target_rows": args.target_rows,
        "task_balance": "exact_property_count_buckets",
        "target_bucket_quotas": {f"{key}p": value for key, value in quotas.items()},
        "cutoff_uint64_by_bucket": {f"{key}p": value for key, value in cutoffs.items()},
        "heldout_hashes": len(frozen), "audit": dict(audit),
        "property_count_distribution": dict(sorted(distribution.items())), "output": files,
    }


def build_edit(args: argparse.Namespace) -> dict[str, object]:
    frozen = heldout_hashes(args.heldout)
    buckets = list(range(1, 8))
    quotas = balanced_quotas(args.target_rows, buckets)
    ranks = {bucket: array.array("Q") for bucket in buckets}
    assignments: Counter[int] = Counter()
    seen: set[int] = set()
    audit: Counter[str] = Counter()
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            audit["input"] += 1
            fields = edit_fields(row)
            if fields is None:
                audit["ineligible"] += 1
                continue
            source, target, active = fields
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            target_hash = hashlib.sha256(target.encode()).hexdigest()
            if source_hash in frozen or target_hash in frozen:
                audit["heldout_overlap"] += 1
                continue
            pair = rank64(f"{source}\n{target}")
            if pair in seen:
                audit["duplicate_pair"] += 1
                continue
            seen.add(pair)
            raw_id = str(row.get("example_id", "") or row.get("sample_id", ""))
            material = f"{args.seed}:{raw_id}:{source}:{target}"
            bucket = assigned_bucket(list(range(1, min(7, len(active)) + 1)), assignments, material)
            assignments[bucket] += 1
            ranks[bucket].append(rank64(material))
            audit["eligible_unique"] += 1
    cutoffs = {bucket: select_cutoff(ranks[bucket], quotas[bucket]) for bucket in buckets}
    assignment_capacity = dict(assignments)
    writer = ShardWriter(args.output_dir, "edit", args.shards)
    seen.clear()
    assignments.clear()
    distribution: Counter[str] = Counter()
    similarity: Counter[str] = Counter()
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            fields = edit_fields(row)
            if fields is None:
                continue
            source, target, active = fields
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            target_hash = hashlib.sha256(target.encode()).hexdigest()
            if source_hash in frozen or target_hash in frozen:
                continue
            pair = rank64(f"{source}\n{target}")
            if pair in seen:
                continue
            seen.add(pair)
            raw_id = str(row.get("example_id", "") or row.get("sample_id", ""))
            material = f"{args.seed}:{raw_id}:{source}:{target}"
            bucket = assigned_bucket(list(range(1, min(7, len(active)) + 1)), assignments, material)
            assignments[bucket] += 1
            if rank64(material) > cutoffs[bucket]:
                continue
            record = edit_record(row, source, target, active, args.seed, bucket)
            writer.write(record)
            distribution[f"{record['property_count']}p"] += 1
            similarity["strict_ge_0.65" if record["source_tanimoto"] >= 0.65 else "relaxed_0.15_0.65"] += 1
            if writer.total == args.target_rows:
                break
    files = writer.close()
    if writer.total != args.target_rows:
        raise AssertionError(f"wrote {writer.total} edit rows, expected {args.target_rows}")
    expected_distribution = {f"{key}p": value for key, value in quotas.items()}
    if dict(sorted(distribution.items())) != expected_distribution:
        raise AssertionError(
            f"edit task distribution is not exactly balanced: {dict(distribution)}"
        )
    return {
        "mode": "edit", "target_rows": args.target_rows,
        "task_balance": "exact_property_count_buckets",
        "target_bucket_quotas": {f"{key}p": value for key, value in quotas.items()},
        "eligible_assignments_by_bucket": {
            f"{key}p": assignment_capacity.get(key, 0) for key in buckets
        },
        "cutoff_uint64_by_bucket": {f"{key}p": value for key, value in cutoffs.items()},
        "heldout_hashes": len(frozen), "audit": dict(audit),
        "property_count_distribution": dict(sorted(distribution.items())),
        "similarity_distribution": dict(similarity), "output": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    denovo = sub.add_parser("de_novo")
    denovo.add_argument("--input", type=Path, action="append", required=True)
    edit = sub.add_parser("edit")
    edit.add_argument("--input-csv", type=Path, required=True)
    for item in (denovo, edit):
        item.add_argument("--output-dir", required=True, type=Path)
        item.add_argument("--target-rows", type=int, default=2_000_000)
        item.add_argument("--shards", type=int, default=128)
        item.add_argument("--seed", type=int, default=24002)
        item.add_argument("--heldout", type=Path, action="append", default=[])
        item.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build_denovo(args) if args.mode == "de_novo" else build_edit(args)
    result.update({"protocol": "molprogram_instruct_release_v1", "seed": args.seed})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in result if k != "output"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

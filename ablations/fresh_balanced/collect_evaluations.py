#!/usr/bin/env python3
"""Collect the preregistered fresh-balanced exposure evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence


SHARED_TASKS = {
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "HBA:increase+MW:increase+QED:decrease",
    "MW:increase",
    "RB:decrease",
}
EDIT_ONLY_TASKS = {
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "HBA:decrease+SA:decrease",
    "QED:increase+SA:decrease",
    "SA:decrease",
}


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash adapter files and their relative names as one frozen checkpoint."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def group_metric(
    buckets: Mapping[str, Mapping[str, object]], tasks: set[str], key: str
) -> float:
    if set(buckets) != SHARED_TASKS | EDIT_ONLY_TASKS:
        raise ValueError("editing summary does not contain the frozen All-10 tasks")
    values = [float(buckets[task][key]) for task in sorted(tasks)]
    return mean(values)


def endpoint(summary: Mapping[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    buckets = summary["edit_buckets"]
    assert isinstance(aggregate, Mapping) and isinstance(buckets, Mapping)

    def editing_group(tasks: set[str]) -> dict[str, float]:
        return {
            "strict_065_macro": group_metric(buckets, tasks, "strict_rate"),
            "valid_macro": group_metric(buckets, tasks, "valid_rate"),
            "property_strict_macro": group_metric(
                buckets, tasks, "property_strict_rate"
            ),
            "source_similarity_macro": group_metric(
                buckets, tasks, "mean_source_similarity"
            ),
        }

    return {
        "de_novo_strict_pooled": float(aggregate["denovo_strict_pooled"]),
        "de_novo_valid_pooled": float(aggregate["denovo_valid_pooled"]),
        "de_novo_strict_2p_7p_macro": float(aggregate["denovo_strict_macro"]),
        "de_novo_valid_2p_7p_macro": float(aggregate["denovo_valid_macro"]),
        "de_novo_property_strict_2p_7p_macro": float(
            aggregate["denovo_property_strict_macro"]
        ),
        "de_novo_property_fraction_2p_7p_macro": float(
            aggregate["denovo_property_fraction_macro"]
        ),
        "editing_all10": {
            "strict_065_macro": float(aggregate["edit_strict_065_macro"]),
            "valid_macro": float(aggregate["edit_valid_macro"]),
            "property_strict_macro": float(aggregate["edit_property_strict_macro"]),
            "source_similarity_macro": float(aggregate["edit_source_similarity_macro"]),
        },
        "editing_shared5": editing_group(SHARED_TASKS),
        "editing_edit_only5": editing_group(EDIT_ONLY_TASKS),
    }


def request_keys(path: Path) -> set[tuple[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        (str(row["task_mode"]), str(row.get("condition_id", row.get("sample_id", ""))))
        for row in rows
    }


def candidate_integrity(
    path: Path, expected_keys: set[tuple[str, str]]
) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(str(row["task_mode"]), str(row["condition_id"])) for row in rows]
    counts = {
        "de_novo": sum(key[0] == "de_novo" for key in keys),
        "edit": sum(key[0] == "edit" for key in keys),
    }
    actual_keys = set(keys)
    return {
        "rows": len(rows),
        "unique_condition_ids_within_mode": len(set(keys)),
        "mode_counts": counts,
        "sha256": sha256(path),
        "matches_frozen_gate": actual_keys == expected_keys,
        "valid": len(rows) == len(set(keys)) == 5440
        and counts == {"de_novo": 440, "edit": 5000}
        and actual_keys == expected_keys,
    }


def collect(
    evaluation_root: Path,
    training_root: Path,
    gate_dir: Path,
    protocol_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    protocol = load(protocol_path)
    gate_manifest = load(gate_dir / "manifest.json")
    gate_hashes_match = (
        sha256(gate_dir / "gate.denovo.jsonl")
        == gate_manifest["de_novo"]["sha256"]
        and sha256(gate_dir / "gate.edit.jsonl")
        == gate_manifest["editing"]["sha256"]
    )
    expected_keys = request_keys(gate_dir / "gate.denovo.jsonl") | request_keys(
        gate_dir / "gate.edit.jsonl"
    )
    if len(expected_keys) != 5440:
        raise ValueError("frozen gate does not contain 5,440 unique requests")
    results: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    integrity: dict[str, object] = {}

    for checkpoint in protocol["checkpoints"]:
        label = str(checkpoint["label"])
        step = int(checkpoint["step"])
        summary_path = evaluation_root / label / "summary.json"
        candidates_path = evaluation_root / label / "candidates.jsonl"
        adapter_path = (
            training_root / "full" / "milestones" / f"checkpoint-{step}" / "adapter"
        )
        milestone_path = adapter_path.parent / "milestone_manifest.json"
        if int(load(milestone_path)["optimizer_step"]) != step:
            raise ValueError(f"milestone step mismatch for {label}")
        summary = load(summary_path)
        if str(summary.get("protocol")) != str(protocol["protocol"]):
            raise ValueError(f"evaluation protocol mismatch for {label}")
        if summary.get("rows") != {"de_novo": 440, "edit": 5000}:
            raise ValueError(f"evaluation row counts mismatch for {label}")
        measured = endpoint(summary)
        candidate_check = candidate_integrity(candidates_path, expected_keys)
        results[label] = {
            "optimizer_step": step,
            "role": checkpoint["role"],
            "metrics": measured,
            "checkpoint_sha256": sha256_tree(adapter_path),
            "adapter_weights_sha256": sha256(
                adapter_path / "adapter_model.safetensors"
            ),
            "summary_sha256": sha256(summary_path),
            "candidates_sha256": candidate_check["sha256"],
        }
        integrity[label] = candidate_check
        csv_rows.append(
            {
                "checkpoint": label,
                "optimizer_step": step,
                "de_novo_strict_pooled": measured["de_novo_strict_pooled"],
                "de_novo_valid_pooled": measured["de_novo_valid_pooled"],
                "de_novo_strict_2p_7p_macro": measured[
                    "de_novo_strict_2p_7p_macro"
                ],
                "de_novo_valid_2p_7p_macro": measured[
                    "de_novo_valid_2p_7p_macro"
                ],
                "de_novo_property_strict_2p_7p_macro": measured[
                    "de_novo_property_strict_2p_7p_macro"
                ],
                "de_novo_property_fraction_2p_7p_macro": measured[
                    "de_novo_property_fraction_2p_7p_macro"
                ],
                "editing_all10_strict_065_macro": measured["editing_all10"][
                    "strict_065_macro"
                ],
                "editing_all10_valid_macro": measured["editing_all10"][
                    "valid_macro"
                ],
                "editing_all10_property_strict_macro": measured["editing_all10"][
                    "property_strict_macro"
                ],
                "editing_all10_source_similarity_macro": measured["editing_all10"][
                    "source_similarity_macro"
                ],
                "editing_shared5_strict_065_macro": measured["editing_shared5"][
                    "strict_065_macro"
                ],
                "editing_shared5_valid_macro": measured["editing_shared5"][
                    "valid_macro"
                ],
                "editing_shared5_property_strict_macro": measured[
                    "editing_shared5"
                ]["property_strict_macro"],
                "editing_shared5_source_similarity_macro": measured[
                    "editing_shared5"
                ]["source_similarity_macro"],
                "editing_edit_only5_strict_065_macro": measured[
                    "editing_edit_only5"
                ]["strict_065_macro"],
                "editing_edit_only5_valid_macro": measured["editing_edit_only5"][
                    "valid_macro"
                ],
                "editing_edit_only5_property_strict_macro": measured[
                    "editing_edit_only5"
                ]["property_strict_macro"],
                "editing_edit_only5_source_similarity_macro": measured[
                    "editing_edit_only5"
                ]["source_similarity_macro"],
            }
        )

    full = results["full"]["metrics"]
    rule = protocol["headline_rule"]
    de_novo_ok = float(full["de_novo_strict_pooled"]) >= float(
        rule["minimum_de_novo_strict_pooled"]
    ) / 100.0
    editing_ok = float(full["editing_all10"]["strict_065_macro"]) >= float(
        rule["minimum_editing_all10_strict_065_macro"]
    ) / 100.0
    complete_integrity = gate_hashes_match and all(
        bool(item["valid"]) for item in integrity.values()
    )
    eligible = de_novo_ok and editing_ok and complete_integrity
    result = {
        "protocol": protocol["protocol"],
        "gate_manifest": gate_manifest,
        "evaluations": results,
        "integrity": {
            "gate_hashes_match": gate_hashes_match,
            "candidate_sets": integrity,
            "all_checks_pass": complete_integrity,
        },
        "decision": {
            "full_checkpoint_only": True,
            "intermediate_milestones_are_descriptive_only": True,
            "de_novo_threshold_pass": de_novo_ok,
            "editing_threshold_pass": editing_ok,
            "full_is_headline_eligible": eligible,
            "headline": (
                "promote_fresh_full" if eligible else "retain_historical_balanced"
            ),
            "conditional_residual_allowed": eligible,
            "safe_grpo_allowed": eligible,
        },
    }
    return result, csv_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result, rows = collect(
        args.evaluation_root, args.training_root, args.gate_dir, args.protocol
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "exposure_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "RESULT_COMPLETE").write_text(sha256(result_path) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

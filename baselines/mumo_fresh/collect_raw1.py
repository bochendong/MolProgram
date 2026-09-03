#!/usr/bin/env python3
"""Validate and collect the official MuMO Raw@1 evaluation summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def number(row: dict[str, str], key: str) -> float | None:
    return float(row[key]) if str(row.get(key, "")).strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--generation-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    task_rows = [row for row in rows if row["external_task_id"] != "all"]
    overall = next(row for row in rows if row["external_task_id"] == "all")
    if len(task_rows) != 10:
        raise ValueError(f"expected ten MuMO task rows, found {len(task_rows)}")
    for row in rows:
        if int(row["candidate_rows"]) != int(row["input_groups"]):
            raise ValueError(f"not Raw@1 for row {row['external_task_id']}")
        if row["success_rate_status"] != "official":
            raise ValueError(f"incomplete oracle coverage for {row['external_task_id']}")
        if number(row, "official_evaluable_rate") != 1.0:
            raise ValueError(f"not fully evaluable: {row['external_task_id']}")
    generation = json.loads(args.generation_summary.read_text(encoding="utf-8"))
    if generation["candidate_budget"] != 1 or generation["conditions"] != 1992:
        raise ValueError("generation summary violates the frozen Raw@1 contract")

    result = {
        "protocol": "mumo_fresh_official_raw1_v1",
        "conditions": generation["conditions"],
        "candidate_budget": 1,
        "raw_at_1": True,
        "tasks": {
            row["external_task_id"]: {
                "split": row["external_task_split"],
                "inputs": int(row["input_groups"]),
                "validity": number(row, "validity"),
                "success_rate": number(row, "success_rate"),
                "similarity_success": number(row, "similarity"),
                "relative_improvement_success": number(row, "relative_improvement"),
                "strict_success_rate": number(row, "strict_success_rate"),
            }
            for row in task_rows
        },
        "overall": {
            "inputs": int(overall["input_groups"]),
            "validity": number(overall, "validity"),
            "success_rate": number(overall, "success_rate"),
            "similarity_success": number(overall, "similarity"),
            "relative_improvement_success": number(overall, "relative_improvement"),
            "strict_success_rate": number(overall, "strict_success_rate"),
        },
        "generation": generation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

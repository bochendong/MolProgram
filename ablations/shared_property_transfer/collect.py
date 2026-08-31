#!/usr/bin/env python3
"""Compare shared-property replay with the matched joint and edit baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


SHARED_TASKS = (
    "RB:decrease",
    "MW:increase",
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "HBA:increase+MW:increase+QED:decrease",
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values)


def shared_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    buckets = summary["edit_buckets"]
    assert isinstance(buckets, Mapping)
    selected = [buckets[task] for task in SHARED_TASKS]
    return {
        "strict_065_macro": mean(item["strict_rate"] for item in selected),
        "relaxed_015_macro": mean(item["relaxed_rate"] for item in selected),
        "valid_macro": mean(item["valid_rate"] for item in selected),
        "property_strict_macro": mean(item["property_strict_rate"] for item in selected),
        "source_similarity_macro": mean(item["mean_source_similarity"] for item in selected),
    }


def summarize(
    candidate: Mapping[str, object],
    joint: Mapping[str, object],
    edit: Mapping[str, object],
) -> dict[str, object]:
    arms = {
        "shared_property_joint": shared_metrics(candidate),
        "naive_joint": shared_metrics(joint),
        "edit_specialist": shared_metrics(edit),
    }
    candidate_metrics = arms["shared_property_joint"]
    edit_metrics = arms["edit_specialist"]
    joint_metrics = arms["naive_joint"]
    deltas = {
        "strict_vs_edit_specialist": (
            candidate_metrics["strict_065_macro"] - edit_metrics["strict_065_macro"]
        ),
        "strict_vs_naive_joint": (
            candidate_metrics["strict_065_macro"] - joint_metrics["strict_065_macro"]
        ),
        "valid_vs_edit_specialist": (
            candidate_metrics["valid_macro"] - edit_metrics["valid_macro"]
        ),
        "source_similarity_vs_edit_specialist": (
            candidate_metrics["source_similarity_macro"]
            - edit_metrics["source_similarity_macro"]
        ),
    }
    positive_transfer = (
        deltas["strict_vs_edit_specialist"] >= 0.02 - 1e-12
        and deltas["valid_vs_edit_specialist"] >= -0.02 - 1e-12
    )
    return {
        "protocol": "shared_property_transfer_v1",
        "primary_endpoint": "Shared-5 editing strict success at similarity >= 0.65",
        "shared_tasks": list(SHARED_TASKS),
        "arms": arms,
        "deltas": deltas,
        "checks": {
            "strict_gain_at_least_2pp_over_edit_specialist": (
                deltas["strict_vs_edit_specialist"] >= 0.02 - 1e-12
            ),
            "validity_within_2pp_of_edit_specialist": (
                deltas["valid_vs_edit_specialist"] >= -0.02 - 1e-12
            ),
        },
        "decision": "positive_transfer" if positive_transfer else "not_supported",
        "full_gate_candidate_aggregate": candidate["aggregate"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--joint-summary", required=True, type=Path)
    parser.add_argument("--edit-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(
        load(args.candidate_summary),
        load(args.joint_summary),
        load(args.edit_summary),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "RESULT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare routed LoRA against the matched vanilla joint and specialists."""

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
EDIT_ONLY_TASKS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "HBA:decrease+SA:decrease",
    "QED:increase+SA:decrease",
    "SA:decrease",
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / len(values)


def editing_subset(summary: Mapping[str, object], tasks: Sequence[str]):
    buckets = summary["edit_buckets"]
    assert isinstance(buckets, Mapping)
    selected = [buckets[task] for task in tasks]
    return {
        "strict_065_macro": mean(item["strict_rate"] for item in selected),
        "valid_macro": mean(item["valid_rate"] for item in selected),
        "property_strict_macro": mean(
            item["property_strict_rate"] for item in selected
        ),
        "source_similarity_macro": mean(
            item["mean_source_similarity"] for item in selected
        ),
    }


def endpoint(summary: Mapping[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    assert isinstance(aggregate, Mapping)
    return {
        "denovo_strict_macro": aggregate["denovo_strict_macro"],
        "denovo_valid_macro": aggregate["denovo_valid_macro"],
        "edit_all10_strict_065_macro": aggregate["edit_strict_065_macro"],
        "edit_all10_valid_macro": aggregate["edit_valid_macro"],
        "edit_shared5": editing_subset(summary, SHARED_TASKS),
        "edit_only5": editing_subset(summary, EDIT_ONLY_TASKS),
    }


def denovo_endpoint(summary: Mapping[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    assert isinstance(aggregate, Mapping)
    return {
        "denovo_strict_macro": aggregate["denovo_strict_macro"],
        "denovo_valid_macro": aggregate["denovo_valid_macro"],
    }


def edit_endpoint(summary: Mapping[str, object]) -> dict[str, object]:
    aggregate = summary["aggregate"]
    assert isinstance(aggregate, Mapping)
    return {
        "edit_all10_strict_065_macro": aggregate["edit_strict_065_macro"],
        "edit_all10_valid_macro": aggregate["edit_valid_macro"],
        "edit_shared5": editing_subset(summary, SHARED_TASKS),
        "edit_only5": editing_subset(summary, EDIT_ONLY_TASKS),
    }


def summarize(candidate, joint, denovo, edit, candidate_train, joint_train):
    arms = {
        "property_program_routed": endpoint(candidate),
        "vanilla_joint": endpoint(joint),
        "denovo_specialist": denovo_endpoint(denovo),
        "edit_specialist": edit_endpoint(edit),
    }
    routed = arms["property_program_routed"]
    vanilla = arms["vanilla_joint"]
    edit_specialist = arms["edit_specialist"]
    denovo_specialist = arms["denovo_specialist"]
    deltas = {
        "edit_only5_strict_vs_vanilla_joint": (
            routed["edit_only5"]["strict_065_macro"]
            - vanilla["edit_only5"]["strict_065_macro"]
        ),
        "edit_only5_strict_vs_edit_specialist": (
            routed["edit_only5"]["strict_065_macro"]
            - edit_specialist["edit_only5"]["strict_065_macro"]
        ),
        "shared5_strict_vs_vanilla_joint": (
            routed["edit_shared5"]["strict_065_macro"]
            - vanilla["edit_shared5"]["strict_065_macro"]
        ),
        "shared5_strict_vs_edit_specialist": (
            routed["edit_shared5"]["strict_065_macro"]
            - edit_specialist["edit_shared5"]["strict_065_macro"]
        ),
        "all10_strict_vs_vanilla_joint": (
            routed["edit_all10_strict_065_macro"]
            - vanilla["edit_all10_strict_065_macro"]
        ),
        "denovo_strict_vs_vanilla_joint": (
            routed["denovo_strict_macro"] - vanilla["denovo_strict_macro"]
        ),
        "denovo_strict_vs_specialist": (
            routed["denovo_strict_macro"] - denovo_specialist["denovo_strict_macro"]
        ),
        "edit_valid_vs_vanilla_joint": (
            routed["edit_all10_valid_macro"] - vanilla["edit_all10_valid_macro"]
        ),
        "denovo_valid_vs_vanilla_joint": (
            routed["denovo_valid_macro"] - vanilla["denovo_valid_macro"]
        ),
    }
    parameter_parity = (
        int(candidate_train["trainable_parameters"])
        == int(joint_train["trainable_parameters"])
        and int(candidate_train["extra_trainable_routing_parameters"]) == 0
    )
    checks = {
        "edit_only_gain_at_least_2pp": (
            deltas["edit_only5_strict_vs_vanilla_joint"] >= 0.02 - 1e-12
        ),
        "shared_strict_within_2pp": (
            deltas["shared5_strict_vs_vanilla_joint"] >= -0.02 - 1e-12
        ),
        "denovo_strict_within_2pp": (
            deltas["denovo_strict_vs_vanilla_joint"] >= -0.02 - 1e-12
        ),
        "edit_validity_within_2pp": (
            deltas["edit_valid_vs_vanilla_joint"] >= -0.02 - 1e-12
        ),
        "denovo_validity_within_2pp": (
            deltas["denovo_valid_vs_vanilla_joint"] >= -0.02 - 1e-12
        ),
        "exact_trainable_parameter_parity": parameter_parity,
    }
    return {
        "protocol": "property_program_routed_lora_10k_v1",
        "primary_endpoint": "Edit-only-5 strict Raw@1 at similarity >= 0.65",
        "arms": arms,
        "deltas": deltas,
        "checks": checks,
        "decision": "supported" if all(checks.values()) else "not_supported",
        "parameter_count": {
            "routed": int(candidate_train["trainable_parameters"]),
            "vanilla_joint": int(joint_train["trainable_parameters"]),
            "extra_routing": int(candidate_train["extra_trainable_routing_parameters"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--joint-summary", required=True, type=Path)
    parser.add_argument("--denovo-summary", required=True, type=Path)
    parser.add_argument("--edit-summary", required=True, type=Path)
    parser.add_argument("--candidate-train", required=True, type=Path)
    parser.add_argument("--joint-train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(
        load(args.candidate_summary),
        load(args.joint_summary),
        load(args.denovo_summary),
        load(args.edit_summary),
        load(args.candidate_train),
        load(args.joint_train),
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

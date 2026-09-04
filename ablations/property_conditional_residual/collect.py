#!/usr/bin/env python3
"""Collect the frozen, conditional, and always-on residual Raw@1 comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
HARD_COLLECT = SCRIPT_DIR.parent / "property_program_routing" / "collect.py"
SPEC = importlib.util.spec_from_file_location("property_routing_collect", HARD_COLLECT)
hard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(hard)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def candidate_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["task_mode"]), str(row["condition_id"])


def raw_identity(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    *,
    subset: str,
) -> dict[str, object]:
    right_by_key = {candidate_key(row): row for row in right}
    checked = 0
    mismatches = []
    for row in left:
        mode = str(row["task_mode"])
        task = str(row.get("task_key", ""))
        include = (
            subset == "inactive"
            and (mode == "de_novo" or task in hard.SHARED_TASKS)
        ) or (subset == "active" and task in hard.EDIT_ONLY_TASKS)
        if not include:
            continue
        checked += 1
        other = right_by_key.get(candidate_key(row))
        if other is None or str(other.get("raw", "")) != str(row.get("raw", "")):
            mismatches.append(candidate_key(row))
    return {
        "subset": subset,
        "checked": checked,
        "mismatch_count": len(mismatches),
        "first_mismatches": [list(item) for item in mismatches[:10]],
        "identical": not mismatches,
    }


def summarize(
    baseline: Mapping[str, object],
    conditional: Mapping[str, object],
    always_on: Mapping[str, object],
    training: Mapping[str, object],
    inactive_identity: Mapping[str, object],
    active_identity: Mapping[str, object],
) -> dict[str, object]:
    base = hard.endpoint(baseline)
    candidate = hard.endpoint(conditional)
    always = hard.endpoint(always_on)

    def deltas(right, left):
        return {
            "denovo_strict_macro": right["denovo_strict_macro"]
            - left["denovo_strict_macro"],
            "denovo_valid_macro": right["denovo_valid_macro"]
            - left["denovo_valid_macro"],
            "edit_all10_strict_065_macro": right["edit_all10_strict_065_macro"]
            - left["edit_all10_strict_065_macro"],
            "edit_all10_valid_macro": right["edit_all10_valid_macro"]
            - left["edit_all10_valid_macro"],
            "edit_shared5_strict_065_macro": right["edit_shared5"]["strict_065_macro"]
            - left["edit_shared5"]["strict_065_macro"],
            "edit_only5_strict_065_macro": right["edit_only5"]["strict_065_macro"]
            - left["edit_only5"]["strict_065_macro"],
        }

    conditional_delta = deltas(candidate, base)
    always_delta = deltas(always, base)
    checks = {
        "edit_only_gain_at_least_2pp": conditional_delta[
            "edit_only5_strict_065_macro"
        ]
        >= 0.02 - 1e-12,
        "edit_all10_nonnegative": conditional_delta[
            "edit_all10_strict_065_macro"
        ]
        >= -1e-12,
        "edit_validity_within_2pp": conditional_delta[
            "edit_all10_valid_macro"
        ]
        >= -0.02 - 1e-12,
        "inactive_raw_outputs_identical": bool(inactive_identity["identical"]),
        "inactive_raw_output_count_is_370": int(inactive_identity["checked"])
        == 370,
        "active_conditional_equals_always_on": bool(active_identity["identical"]),
        "active_raw_output_count_is_250": int(active_identity["checked"]) == 250,
        "shared_weights_exactly_frozen": float(
            training["shared_slice_max_abs_delta"]
        )
        == 0.0,
    }
    return {
        "protocol": "property_conditional_residual_rank4_pilot_v1",
        "primary_endpoint": "Edit-only-5 strict Raw@1 at similarity >= 0.65",
        "arms": {
            "frozen_shared": base,
            "conditional_residual": candidate,
            "always_on_residual": always,
        },
        "deltas_conditional_minus_frozen": conditional_delta,
        "deltas_always_on_minus_frozen": always_delta,
        "identity_checks": {
            "frozen_vs_conditional_inactive": dict(inactive_identity),
            "conditional_vs_always_on_active": dict(active_identity),
        },
        "checks": checks,
        "decision": "supported" if all(checks.values()) else "not_supported",
        "confirmatory_claim_allowed": False,
        "parameter_count": {
            "expanded_adapter": int(training["trainable_tensor_parameters"]),
            "effectively_updated_residual": int(
                training["effectively_updated_residual_parameters"]
            ),
            "shared_rank": int(training["shared_rank"]),
            "residual_rank": int(training["residual_rank"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--conditional-summary", required=True, type=Path)
    parser.add_argument("--always-on-summary", required=True, type=Path)
    parser.add_argument("--baseline-candidates", required=True, type=Path)
    parser.add_argument("--conditional-candidates", required=True, type=Path)
    parser.add_argument("--always-on-candidates", required=True, type=Path)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    baseline_candidates = read_jsonl(args.baseline_candidates)
    conditional_candidates = read_jsonl(args.conditional_candidates)
    always_candidates = read_jsonl(args.always_on_candidates)
    result = summarize(
        load(args.baseline_summary),
        load(args.conditional_summary),
        load(args.always_on_summary),
        load(args.training_summary),
        raw_identity(baseline_candidates, conditional_candidates, subset="inactive"),
        raw_identity(conditional_candidates, always_candidates, subset="active"),
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

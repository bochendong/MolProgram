#!/usr/bin/env python3
"""Collect the post-hoc dense-inference diagnostic for the routed adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from collect import endpoint, load  # noqa: E402


def compact(summary: Mapping[str, object]) -> dict[str, float]:
    values = endpoint(summary)
    return {
        "denovo_strict_macro": float(values["denovo_strict_macro"]),
        "denovo_valid_macro": float(values["denovo_valid_macro"]),
        "edit_all10_strict_065_macro": float(values["edit_all10_strict_065_macro"]),
        "edit_all10_valid_macro": float(values["edit_all10_valid_macro"]),
        "edit_shared5_strict_065_macro": float(
            values["edit_shared5"]["strict_065_macro"]
        ),
        "edit_only5_strict_065_macro": float(
            values["edit_only5"]["strict_065_macro"]
        ),
    }


def subtract(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {name: left[name] - right[name] for name in left}


def summarize(dense, hard, vanilla) -> dict[str, object]:
    arms = {
        "dense_inference": compact(dense),
        "hard_routed_inference": compact(hard),
        "vanilla_joint": compact(vanilla),
    }
    dense_values = arms["dense_inference"]
    hard_values = arms["hard_routed_inference"]
    vanilla_values = arms["vanilla_joint"]
    dense_minus_hard = subtract(dense_values, hard_values)
    dense_minus_vanilla = subtract(dense_values, vanilla_values)
    return {
        "protocol": "property_program_routed_lora_10k_dense_inference_diagnostic_v1",
        "status": "post_hoc_mechanism_diagnostic",
        "arms": arms,
        "deltas": {
            "dense_minus_hard": dense_minus_hard,
            "dense_minus_vanilla": dense_minus_vanilla,
        },
        "diagnostic_checks": {
            "denovo_recovers_at_least_2pp_vs_hard": (
                dense_minus_hard["denovo_strict_macro"] >= 0.02
            ),
            "edit_only_recovers_at_least_2pp_vs_hard": (
                dense_minus_hard["edit_only5_strict_065_macro"] >= 0.02
            ),
            "denovo_within_2pp_of_vanilla": (
                dense_minus_vanilla["denovo_strict_macro"] >= -0.02
            ),
            "edit_only_within_2pp_of_vanilla": (
                dense_minus_vanilla["edit_only5_strict_065_macro"] >= -0.02
            ),
            "shared_gain_at_least_2pp_vs_vanilla": (
                dense_minus_vanilla["edit_shared5_strict_065_macro"] >= 0.02
            ),
        },
        "confirmatory_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-summary", required=True, type=Path)
    parser.add_argument("--hard-summary", required=True, type=Path)
    parser.add_argument("--vanilla-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(
        load(args.dense_summary), load(args.hard_summary), load(args.vanilla_summary)
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

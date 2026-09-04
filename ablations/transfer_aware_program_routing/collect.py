#!/usr/bin/env python3
"""Compare transfer-aware routing with its byte-identical dense LoRA control."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
HARD_ROUTING = SCRIPT_DIR.parent / "property_program_routing" / "collect.py"
SPEC = importlib.util.spec_from_file_location("hard_routing_collect", HARD_ROUTING)
hard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(hard)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def summarize(
    routed: Mapping[str, object],
    dense: Mapping[str, object],
    routed_train: Mapping[str, object],
    dense_train: Mapping[str, object],
    graph: Mapping[str, object],
) -> dict[str, object]:
    candidate = hard.endpoint(routed)
    control = hard.endpoint(dense)
    deltas = {
        "denovo_strict_macro": candidate["denovo_strict_macro"]
        - control["denovo_strict_macro"],
        "denovo_valid_macro": candidate["denovo_valid_macro"]
        - control["denovo_valid_macro"],
        "edit_all10_strict_065_macro": candidate["edit_all10_strict_065_macro"]
        - control["edit_all10_strict_065_macro"],
        "edit_all10_valid_macro": candidate["edit_all10_valid_macro"]
        - control["edit_all10_valid_macro"],
        "edit_shared5_strict_065_macro": candidate["edit_shared5"]["strict_065_macro"]
        - control["edit_shared5"]["strict_065_macro"],
        "edit_only5_strict_065_macro": candidate["edit_only5"]["strict_065_macro"]
        - control["edit_only5"]["strict_065_macro"],
    }
    parameter_parity = (
        int(routed_train["trainable_parameters"])
        == int(dense_train["trainable_parameters"])
        and int(routed_train["extra_trainable_routing_parameters"]) == 0
        and int(dense_train["extra_trainable_routing_parameters"]) == 0
    )
    checks = {
        "edit_only_gain_at_least_2pp": deltas["edit_only5_strict_065_macro"] >= 0.02,
        "edit_all10_nonnegative": deltas["edit_all10_strict_065_macro"] >= 0.0,
        "shared_within_2pp": deltas["edit_shared5_strict_065_macro"] >= -0.02,
        "denovo_within_2pp": deltas["denovo_strict_macro"] >= -0.02,
        "both_validities_within_2pp": (
            deltas["denovo_valid_macro"] >= -0.02
            and deltas["edit_all10_valid_macro"] >= -0.02
        ),
        "parameter_parity": parameter_parity,
    }
    return {
        "protocol": "transfer_aware_program_routing_10k_pilot_v1",
        "primary_endpoint": "Edit-only-5 strict Raw@1 at similarity >= 0.65",
        "arms": {"transfer_aware": candidate, "matched_dense": control},
        "deltas_transfer_aware_minus_dense": deltas,
        "checks": checks,
        "decision": "supported" if all(checks.values()) else "not_supported",
        "confirmatory_claim_allowed": False,
        "graph_nodes": list(graph["nodes"]),
        "graph_eigenvalues": list(graph["eigenvalues"]),
        "parameter_count": {
            "transfer_aware": int(routed_train["trainable_parameters"]),
            "matched_dense": int(dense_train["trainable_parameters"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-summary", required=True, type=Path)
    parser.add_argument("--dense-summary", required=True, type=Path)
    parser.add_argument("--routed-train", required=True, type=Path)
    parser.add_argument("--dense-train", required=True, type=Path)
    parser.add_argument("--transfer-graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(
        load(args.routed_summary),
        load(args.dense_summary),
        load(args.routed_train),
        load(args.dense_train),
        load(args.transfer_graph),
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

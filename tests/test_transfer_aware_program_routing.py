from __future__ import annotations

import math

import numpy as np
import pytest

from molprogram import program_routing, transfer_graph


def row(mode: str, *properties: str):
    return {
        "task_mode": mode,
        "condition_program": [
            {"property": prop, "goal": "increase"} for prop in properties
        ],
    }


def cosine(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_signed_spectral_layout_shares_aligned_nodes_and_splits_conflicts():
    signatures = {
        "de_novo:MW": [1.0, 0.1, 0.0],
        "edit:MW": [0.9, 0.2, 0.0],
        "edit:SA": [-1.0, -0.1, 0.1],
        "edit:GSK3B": [-0.8, 0.0, 0.2],
    }
    layout, evidence = transfer_graph.compile_signed_spectral_layout(
        signatures, rank=6, common_ranks=2, inactive_floor=0.2
    )
    program_routing.validate_layout(layout)

    denovo_mw = program_routing.route_values(row("de_novo", "MW"), layout)
    edit_mw = program_routing.route_values(row("edit", "MW"), layout)
    edit_sa = program_routing.route_values(row("edit", "SA"), layout)

    assert cosine(denovo_mw[2:], edit_mw[2:]) > cosine(denovo_mw[2:], edit_sa[2:])
    assert all(value > 0.0 for value in denovo_mw[2:])
    assert sum(value * value for value in edit_sa) == pytest.approx(6.0)
    assert evidence["nodes"] == sorted(signatures)


def test_multi_property_route_is_a_soft_union_with_fixed_energy():
    signatures = {
        "de_novo:MW": [1.0, 0.0, 0.0],
        "edit:MW": [0.8, 0.1, 0.0],
        "edit:SA": [0.0, 1.0, 0.2],
        "edit:QED": [0.7, 0.6, 0.0],
    }
    layout, _ = transfer_graph.compile_signed_spectral_layout(
        signatures, rank=6, common_ranks=2, inactive_floor=0.25
    )
    mw = program_routing.route_values(row("edit", "MW"), layout)
    sa = program_routing.route_values(row("edit", "SA"), layout)
    combined = program_routing.route_values(row("edit", "MW", "SA"), layout)

    assert all(value > 0.0 for value in combined)
    assert math.isclose(sum(value * value for value in combined), 6.0)
    assert combined != mw
    assert combined != sa

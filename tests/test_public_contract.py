import json

import pytest

pytest.importorskip("rdkit")

from molprogram import protocol
from molprogram.rewards import INELIGIBLE_FLOOR, hard_boundary_reward
from molprogram.scoring import property_count, score_response


def request(source, conditions, mode):
    return {
        "task_mode": mode,
        "messages": [
            {"role": "system", "content": protocol.SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"source": source, "conditions": conditions}),
            },
        ],
    }


def test_de_novo_response_is_strict_and_target_blind():
    row = request(
        "<EMPTY>",
        [{"property": "MW", "goal": {"around": 46.069}}],
        "de_novo",
    )
    raw = protocol.response("CCO", "de_novo")
    reward, details = score_response(row, raw)
    assert property_count(row) == 1
    assert reward > 0
    assert details["valid"] is True
    assert details["property_strict"] is True
    assert "target_smiles" not in json.dumps(row)


def test_plan_must_match_mode():
    parsed = protocol.parse_response(
        '{"plan":"MODIFY","smiles":"CCO"}', "de_novo"
    )
    assert parsed["valid"] is False


def test_edit_copy_is_exposed_to_reward():
    row = request(
        "CCO",
        [{"property": "QED", "goal": "preserve"}],
        "edit",
    )
    _, details = score_response(row, protocol.response("CCO", "edit"))
    assert details["copy"] is True
    assert details["source_similarity"] == pytest.approx(1.0)


def test_hard_boundary_blocks_ineligible_edits():
    details = {
        "valid": True,
        "copy": False,
        "source_similarity": 0.64,
        "canonical": True,
        "mean_satisfaction": 1.0,
        "bottleneck": 1.0,
        "property_fraction": 1.0,
        "property_strict": True,
        "strict": False,
    }
    assert hard_boundary_reward({}, details, "edit") == INELIGIBLE_FLOOR

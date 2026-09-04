import json

import pytest

pytest.importorskip("rdkit")

from molprogram import protocol
from molprogram.rewards import INELIGIBLE_FLOOR, hard_boundary_reward
from molprogram.scoring import (
    PROPERTY_NORMALIZERS,
    STRICT_TOLERANCE,
    molecular_properties,
    property_count,
    score_response,
)


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


def test_de_novo_uses_frozen_paper_tolerances():
    ethanol_mw = molecular_properties("CCO")["MW"]
    within = request(
        "<EMPTY>",
        [{"property": "MW", "goal": {"around": ethanol_mw + 34.9}}],
        "de_novo",
    )
    outside = request(
        "<EMPTY>",
        [{"property": "MW", "goal": {"around": ethanol_mw + 35.1}}],
        "de_novo",
    )
    raw = protocol.response("CCO", "de_novo")
    assert score_response(within, raw)[1]["strict"] is True
    assert score_response(outside, raw)[1]["strict"] is False
    assert STRICT_TOLERANCE == {
        "MW": 35.0,
        "LogP": 1.0,
        "QED": 0.10,
        "TPSA": 20.0,
        "HBD": 1.0,
        "HBA": 1.0,
        "RB": 1.0,
    }


def test_reward_scales_match_frozen_joint_sweep():
    assert PROPERTY_NORMALIZERS["MW"] == 500.0
    assert PROPERTY_NORMALIZERS["LogP"] == 6.0
    assert PROPERTY_NORMALIZERS["HBA"] == 12.0
    assert PROPERTY_NORMALIZERS["SA"] == 8.0


def test_descriptors_match_paper_lipinski_definitions():
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    molecule = Chem.MolFromSmiles("CCN(C)C(=O)NCCO")
    scores = molecular_properties("CCN(C)C(=O)NCCO")
    assert scores["HBD"] == Lipinski.NumHDonors(molecule)
    assert scores["HBA"] == Lipinski.NumHAcceptors(molecule)
    assert scores["RB"] == Lipinski.NumRotatableBonds(molecule)


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

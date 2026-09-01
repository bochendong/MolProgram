import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from molprogram import protocol
from molprogram.rewards import INELIGIBLE_FLOOR


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RL = runpy.run_path(str(ROOT / "scripts" / "train_rl.py"))


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


def test_training_uses_hard_boundary_for_editing_by_default():
    row = request("CCO", [{"property": "QED", "goal": "preserve"}], "edit")
    reward, details = TRAIN_RL["reward_response"](
        row, protocol.response("CCO", "edit")
    )
    assert details["copy"] is True
    assert reward == INELIGIBLE_FLOOR


def test_soft_editing_reward_remains_an_explicit_ablation():
    row = request("CCO", [{"property": "QED", "goal": "preserve"}], "edit")
    reward, _ = TRAIN_RL["reward_response"](
        row, protocol.response("CCO", "edit"), "soft"
    )
    assert reward > INELIGIBLE_FLOOR

import json
import runpy
from pathlib import Path

import pytest

from molprogram.support_audit import (
    DEFAULT_THRESHOLDS,
    summarize_support,
    validate_rl_authorization,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = runpy.run_path(
    str(ROOT / "scripts" / "audit_editing_reward_support.py")
)


def candidate(*, strict: bool, hard_reward: float, soft_reward: float = 0.0):
    return {
        "valid": True,
        "copy": False,
        "source_feasible": strict,
        "property_strict": strict,
        "strict": strict,
        "source_similarity": 0.8 if strict else 0.3,
        "hard_reward": hard_reward,
        "soft_reward": soft_reward,
    }


def groups_with_support(*, hard_ranks_strict: bool = True):
    groups = []
    for index in range(10):
        strict_reward = 5.0 if hard_ranks_strict else -4.0
        failed_reward = -4.0 if hard_ranks_strict else 5.0
        groups.append({
            "example_id": f"example-{index}",
            "task_key": f"task-{index}",
            "candidates": [
                candidate(strict=True, hard_reward=strict_reward, soft_reward=3.0),
                candidate(strict=False, hard_reward=failed_reward, soft_reward=1.0),
            ],
        })
    return groups


def test_support_gate_promotes_only_informative_well_ranked_groups():
    summary = summarize_support(groups_with_support())
    assert summary["gate"]["decision"] == "PROCEED_TO_SMALL_ONLINE_RL_PILOT"
    assert summary["aggregate"]["group"]["mixed_strict_group_rate"] == 1.0
    assert summary["gate"]["supported_tasks"] == 10


def test_support_gate_stops_when_policy_has_no_strict_candidates():
    groups = [{
        "example_id": "no-support",
        "task_key": "task-a",
        "candidates": [candidate(strict=False, hard_reward=-4.0) for _ in range(4)],
    }]
    summary = summarize_support(groups)
    assert summary["gate"]["decision"] == "DO_NOT_RUN_ONLINE_RL_SUPPORT_TOO_LOW"
    assert "strict_any_at_k" in summary["gate"]["failed_checks"]


def test_support_gate_detects_reward_ranking_failure():
    summary = summarize_support(groups_with_support(hard_ranks_strict=False))
    assert summary["gate"]["decision"] == "REPAIR_REWARD_BEFORE_ONLINE_RL"
    assert summary["gate"]["checks"]["hard_reward_ranking"] is False


def test_groups_jsonl_cli_writes_machine_and_human_reports(tmp_path):
    groups_path = tmp_path / "groups.jsonl"
    groups_path.write_text(
        "".join(json.dumps(group) + "\n" for group in groups_with_support())
    )
    output_dir = tmp_path / "report"
    assert AUDIT_SCRIPT["main"]([
        "--groups-jsonl", str(groups_path),
        "--output-dir", str(output_dir),
    ]) == 0
    summary = json.loads((output_dir / "support_report.json").read_text())
    assert summary["gate"]["decision"] == "PROCEED_TO_SMALL_ONLINE_RL_PILOT"
    assert "Per-task strict support" in (output_dir / "support_report.md").read_text()
    assert (output_dir / "AUDIT_COMPLETE").is_file()


def test_preregistered_thresholds_match_runtime_gate():
    preregistration = json.loads((
        ROOT / "audits" / "editing_reward_support" / "preregistration.json"
    ).read_text())
    assert preregistration["thresholds"] == DEFAULT_THRESHOLDS


def test_training_authorization_accepts_only_a_passed_complete_audit():
    summary = summarize_support(groups_with_support())
    assert validate_rl_authorization(summary) == "PROCEED_TO_SMALL_ONLINE_RL_PILOT"
    summary["gate"]["decision"] = "BUILD_SUPPORT_BEFORE_ONLINE_RL"
    with pytest.raises(ValueError, match="does not authorize"):
        validate_rl_authorization(summary)


def test_training_authorization_rejects_threshold_drift():
    summary = summarize_support(groups_with_support())
    summary["gate"]["thresholds"]["strict_any_at_k_min"] = 0.0
    with pytest.raises(ValueError, match="frozen thresholds"):
        validate_rl_authorization(summary)

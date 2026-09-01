"""Reward-support diagnostics for source-conditioned editing policies."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence


DEFAULT_THRESHOLDS = {
    "strict_any_at_k_min": 0.05,
    "mixed_strict_group_rate_min": 0.20,
    "hard_informative_group_rate_min": 0.20,
    "hard_top_strict_rate_min": 0.70,
    "supported_task_fraction_min": 0.70,
    "task_count_min": 10,
}
AUTHORIZED_RL_DECISION = "PROCEED_TO_SMALL_ONLINE_RL_PILOT"


def validate_rl_authorization(report: Mapping[str, object]) -> str:
    """Reject editing RL unless a complete target-blind support audit passed."""
    if str(report.get("protocol", "")) != "molprogram_editing_reward_support_audit_v1":
        raise ValueError("editing support report has an unsupported protocol")
    if report.get("target_smiles_access") is not False:
        raise ValueError("editing support audit must record target_smiles_access=false")
    gate = report.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("editing support report is missing its gate")
    if dict(gate.get("thresholds", {})) != DEFAULT_THRESHOLDS:
        raise ValueError("editing support report does not use frozen thresholds")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(checks.values()):
        raise ValueError("editing support report has failed or incomplete checks")
    if list(gate.get("failed_checks", [])):
        raise ValueError("editing support report records failed checks")
    decision = str(gate.get("decision", ""))
    if decision != AUTHORIZED_RL_DECISION:
        raise ValueError(f"editing support gate does not authorize online RL: {decision}")
    if int(gate.get("task_count", 0)) < int(DEFAULT_THRESHOLDS["task_count_min"]):
        raise ValueError("editing support audit does not cover every registered task")
    return decision


def mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def _candidate_records(groups: Sequence[Mapping[str, object]]):
    for group in groups:
        for candidate in list(group.get("candidates", [])):
            if isinstance(candidate, Mapping):
                yield candidate


def summarize_partition(groups: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize candidate and group support without reading target molecules."""
    groups = list(groups)
    candidates = list(_candidate_records(groups))
    if not groups or not candidates:
        raise ValueError("support audit requires non-empty groups and candidates")
    group_sizes = [len(list(group.get("candidates", []))) for group in groups]
    if min(group_sizes) <= 0:
        raise ValueError("every support-audit group must contain candidates")

    group_records: list[dict[str, object]] = []
    top_soft_supported = 0
    top_hard_supported = 0
    supported_groups = 0
    for group in groups:
        items = [
            item for item in list(group.get("candidates", []))
            if isinstance(item, Mapping)
        ]
        strict_count = sum(bool(item.get("strict")) for item in items)
        hard_rewards = [float(item.get("hard_reward", 0.0)) for item in items]
        soft_rewards = [float(item.get("soft_reward", 0.0)) for item in items]
        hard_informative = max(hard_rewards) - min(hard_rewards) > 1e-12
        soft_informative = max(soft_rewards) - min(soft_rewards) > 1e-12
        if strict_count:
            supported_groups += 1
            max_soft = max(soft_rewards)
            max_hard = max(hard_rewards)
            top_soft_supported += int(any(
                bool(item.get("strict"))
                and abs(float(item.get("soft_reward", 0.0)) - max_soft) <= 1e-12
                for item in items
            ))
            top_hard_supported += int(any(
                bool(item.get("strict"))
                and abs(float(item.get("hard_reward", 0.0)) - max_hard) <= 1e-12
                for item in items
            ))
        group_records.append({
            "strict_any": strict_count > 0,
            "mixed_strict": 0 < strict_count < len(items),
            "hard_informative": hard_informative,
            "soft_informative": soft_informative,
        })

    similarities = [
        float(item["source_similarity"])
        for item in candidates if item.get("source_similarity") is not None
    ]
    return {
        "groups": len(groups),
        "candidates": len(candidates),
        "group_size_min": min(group_sizes),
        "group_size_max": max(group_sizes),
        "candidate": {
            "valid_rate": mean(bool(item.get("valid")) for item in candidates),
            "noncopy_rate": mean(not bool(item.get("copy")) for item in candidates),
            "source_feasible_rate": mean(
                bool(item.get("source_feasible")) for item in candidates
            ),
            "property_strict_rate": mean(
                bool(item.get("property_strict")) for item in candidates
            ),
            "strict_rate": mean(bool(item.get("strict")) for item in candidates),
            "mean_source_similarity": mean(similarities) if similarities else None,
        },
        "group": {
            "strict_any_at_k": mean(item["strict_any"] for item in group_records),
            "mixed_strict_group_rate": mean(
                item["mixed_strict"] for item in group_records
            ),
            "hard_informative_group_rate": mean(
                item["hard_informative"] for item in group_records
            ),
            "soft_informative_group_rate": mean(
                item["soft_informative"] for item in group_records
            ),
            "soft_top_strict_rate_given_support": (
                top_soft_supported / supported_groups if supported_groups else None
            ),
            "hard_top_strict_rate_given_support": (
                top_hard_supported / supported_groups if supported_groups else None
            ),
        },
    }


def summarize_support(
    groups: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Return per-task support, aggregate diagnostics, and an RL go/no-go gate."""
    thresholds = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    groups = list(groups)
    by_task: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for group in groups:
        task = str(group.get("task_key", "") or "")
        if not task:
            raise ValueError("support-audit group is missing task_key")
        by_task[task].append(group)
    aggregate = summarize_partition(groups)
    tasks = {
        task: summarize_partition(task_groups)
        for task, task_groups in sorted(by_task.items())
    }

    supported_tasks = sum(
        float(task["group"]["strict_any_at_k"])
        >= float(thresholds["strict_any_at_k_min"])
        for task in tasks.values()
    )
    supported_task_fraction = supported_tasks / max(len(tasks), 1)
    group = aggregate["group"]
    ranking = group["hard_top_strict_rate_given_support"]
    checks = {
        "strict_any_at_k": (
            float(group["strict_any_at_k"])
            >= float(thresholds["strict_any_at_k_min"])
        ),
        "mixed_strict_groups": (
            float(group["mixed_strict_group_rate"])
            >= float(thresholds["mixed_strict_group_rate_min"])
        ),
        "hard_informative_groups": (
            float(group["hard_informative_group_rate"])
            >= float(thresholds["hard_informative_group_rate_min"])
        ),
        "hard_reward_ranking": (
            ranking is not None
            and float(ranking) >= float(thresholds["hard_top_strict_rate_min"])
        ),
        "task_coverage": (
            supported_task_fraction
            >= float(thresholds["supported_task_fraction_min"])
        ),
        "task_count": len(tasks) >= int(thresholds["task_count_min"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if not checks["strict_any_at_k"] or not checks["hard_informative_groups"]:
        decision = "DO_NOT_RUN_ONLINE_RL_SUPPORT_TOO_LOW"
    elif not checks["hard_reward_ranking"]:
        decision = "REPAIR_REWARD_BEFORE_ONLINE_RL"
    elif failed:
        decision = "BUILD_SUPPORT_BEFORE_ONLINE_RL"
    else:
        decision = "PROCEED_TO_SMALL_ONLINE_RL_PILOT"
    return {
        "protocol": "molprogram_editing_reward_support_audit_v1",
        "target_smiles_access": False,
        "aggregate": aggregate,
        "tasks": tasks,
        "gate": {
            "decision": decision,
            "thresholds": thresholds,
            "checks": checks,
            "failed_checks": failed,
            "supported_tasks": supported_tasks,
            "task_count": len(tasks),
            "supported_task_fraction": supported_task_fraction,
        },
    }

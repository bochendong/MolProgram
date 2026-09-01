"""Pure contracts for mode-balanced, target-blind Raw@1 GRPO."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Mapping, Sequence

from . import protocol
from .scoring import property_count, score_response


DE_NOVO_BUCKETS = tuple(f"de_novo:{count}p" for count in range(2, 8))
EDIT_BUCKETS = tuple(
    f"edit:{task}" for task in sorted(protocol.TABLE1_TASK_KEYS.values())
)
CHANNEL_WEIGHTS = {
    "de_novo": {
        "validity": 0.50,
        "canonical": 0.10,
        "property_mean": 0.75,
        "property_bottleneck": 1.00,
        "property_strict": 1.00,
    },
    "edit": {
        "validity": 0.50,
        "canonical": 0.10,
        "property_mean": 0.75,
        "property_bottleneck": 1.00,
        "property_strict": 1.00,
        "source_aligned": 0.50,
        "source_relaxed": 0.25,
        "source_strict": 0.50,
        "relaxed_success": 0.50,
        "strict_success": 1.00,
        "noncopy": 0.25,
    },
}


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get(
        "example_id", row.get("sample_id", row.get("condition_id", ""))
    )
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def balanced_bucket(row: Mapping[str, object]) -> str:
    mode = str(row.get("task_mode", ""))
    if mode == "de_novo":
        count = property_count(row)
        return f"de_novo:{count}p" if 2 <= count <= 7 else ""
    if mode == "edit":
        task = str(row.get("task_key", ""))
        return f"edit:{task}" if f"edit:{task}" in EDIT_BUCKETS else ""
    return ""


def select_balanced_pairs(
    rows: Sequence[dict[str, object]], pairs: int, seed: int
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Select paired modes with equal exposure inside all 6+10 task buckets."""
    if pairs <= 0 or pairs % 30:
        raise ValueError("paired steps must be a positive multiple of 30")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = balanced_bucket(row)
        if bucket:
            grouped[bucket].append(row)

    selected_modes: list[list[dict[str, object]]] = []
    for offset, buckets in enumerate((DE_NOVO_BUCKETS, EDIT_BUCKETS)):
        per_bucket = pairs // len(buckets)
        selected: list[dict[str, object]] = []
        for bucket in buckets:
            candidates = sorted(
                grouped[bucket], key=lambda row: stable_key(row, seed + offset)
            )
            if len(candidates) < per_bucket:
                raise ValueError(
                    f"insufficient rows for {bucket}: {len(candidates)} < {per_bucket}"
                )
            selected.extend(candidates[:per_bucket])
        random.Random(seed + 10 + offset).shuffle(selected)
        selected_modes.append(selected)
    if any(len(rows_for_mode) != pairs for rows_for_mode in selected_modes):
        raise AssertionError("balanced selector produced unequal mode exposure")
    return list(zip(selected_modes[0], selected_modes[1]))


def reward_channels(
    row: Mapping[str, object], raw: str
) -> tuple[dict[str, float], dict[str, object]]:
    """Return prompt-visible candidate channels without a best-of-K objective."""
    mode = str(row.get("task_mode", ""))
    if mode not in CHANNEL_WEIGHTS:
        raise ValueError(f"unsupported task mode: {mode}")
    _, details = score_response(row, raw)
    channels = {name: 0.0 for name in CHANNEL_WEIGHTS[mode]}
    if not bool(details.get("valid")):
        return channels, details

    channels.update(
        {
            "validity": 1.0,
            "canonical": float(bool(details.get("canonical"))),
            "property_mean": float(details.get("mean_satisfaction", 0.0)),
            "property_bottleneck": float(details.get("bottleneck", 0.0)),
            "property_strict": float(bool(details.get("property_strict"))),
        }
    )
    if mode == "edit":
        similarity = float(details.get("source_similarity") or 0.0)
        source_aligned = 0.5 * (math.tanh((similarity - 0.65) / 0.15) + 1.0)
        channels.update(
            {
                "source_aligned": source_aligned,
                "source_relaxed": float(similarity >= 0.15),
                "source_strict": float(similarity >= 0.65),
                "relaxed_success": float(bool(details.get("relaxed"))),
                "strict_success": float(bool(details.get("strict"))),
                "noncopy": float(not bool(details.get("copy"))),
            }
        )
    return channels, details


def zscores(values: Sequence[float], clip: float = 3.0) -> list[float]:
    center = sum(float(value) for value in values) / max(len(values), 1)
    variance = sum(
        (float(value) - center) ** 2 for value in values
    ) / max(len(values), 1)
    if variance < 1e-12:
        return [0.0 for _ in values]
    scale = variance**0.5
    return [max(-clip, min(clip, (float(value) - center) / scale)) for value in values]


def decoupled_advantages(
    channel_rows: Sequence[Mapping[str, float]], weights: Mapping[str, float]
) -> tuple[list[float], dict[str, object]]:
    """Normalize channels within one prompt group before combining them."""
    combined = [0.0 for _ in channel_rows]
    active: list[str] = []
    for name, weight in weights.items():
        normalized = zscores([float(row.get(name, 0.0)) for row in channel_rows])
        if any(abs(value) > 0.0 for value in normalized):
            active.append(name)
        for index, value in enumerate(normalized):
            combined[index] += float(weight) * value
    advantages = zscores(combined)
    return advantages, {
        "active_channels": active,
        "active_channel_count": len(active),
        "zero_signal": not any(abs(value) > 0.0 for value in advantages),
    }


def scalar_rewards(
    channel_rows: Sequence[Mapping[str, float]], weights: Mapping[str, float]
) -> list[float]:
    return [
        sum(float(weights[name]) * float(row.get(name, 0.0)) for name in weights)
        for row in channel_rows
    ]


def equal_norm_bisector(first, second):
    """Merge two task gradients without letting either norm dominate."""
    import torch

    if len(first) != len(second):
        raise ValueError("gradient lists must have equal length")
    first_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in first))
    second_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in second))
    first_safe = first_norm.clamp_min(1e-12)
    second_safe = second_norm.clamp_min(1e-12)
    cosine = sum(
        (left.float() * right.float()).sum()
        for left, right in zip(first, second)
    ) / (first_safe * second_safe)
    scale = 0.5 * (first_norm + second_norm)
    merged = [
        0.5 * scale * (left / first_safe + right / second_safe)
        for left, right in zip(first, second)
    ]
    merged_norm = torch.sqrt(sum(value.float().pow(2).sum() for value in merged))
    dot_first = sum(
        (value.float() * source.float()).sum()
        for value, source in zip(merged, first)
    )
    dot_second = sum(
        (value.float() * source.float()).sum()
        for value, source in zip(merged, second)
    )
    return merged, {
        "gradient_cosine": float(cosine),
        "gradient_conflict": bool(float(cosine) < 0.0),
        "denovo_gradient_norm": float(first_norm),
        "edit_gradient_norm": float(second_norm),
        "merged_gradient_norm": float(merged_norm),
        "merged_dot_denovo": float(dot_first),
        "merged_dot_edit": float(dot_second),
        "common_descent": bool(
            float(dot_first) >= -1e-8 and float(dot_second) >= -1e-8
        ),
    }


def group_record(
    row: Mapping[str, object],
    rewards: Sequence[float],
    details: Sequence[Mapping[str, object]],
    advantage_record: Mapping[str, object],
) -> dict[str, object]:
    mean_reward = sum(float(value) for value in rewards) / max(len(rewards), 1)
    return {
        "example_id": row.get("example_id", row.get("sample_id", "")),
        "bucket": balanced_bucket(row),
        "mean_reward": mean_reward,
        "reward_std": (
            sum((float(value) - mean_reward) ** 2 for value in rewards)
            / max(len(rewards), 1)
        ) ** 0.5,
        "valid_fraction": sum(bool(item.get("valid")) for item in details)
        / max(len(details), 1),
        "strict_fraction": sum(bool(item.get("strict")) for item in details)
        / max(len(details), 1),
        "relaxed_fraction": sum(bool(item.get("relaxed")) for item in details)
        / max(len(details), 1),
        "advantage": dict(advantage_record),
    }

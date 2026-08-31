"""Lexicographic hard-boundary reward for source-conditioned editing RL."""

from __future__ import annotations

from typing import Mapping


INELIGIBLE_FLOOR = -4.0
STRICT_SIMILARITY = 0.65


def hard_boundary_reward(
    channels: Mapping[str, float], details: Mapping[str, object], mode: str
) -> float:
    """Only source-feasible, non-copy edits can rise above the reward floor."""
    del channels
    if mode != "edit":
        raise ValueError("hard-boundary reward is defined only for editing")
    similarity = float(details.get("source_similarity") or 0.0)
    eligible = (
        bool(details.get("valid"))
        and not bool(details.get("copy"))
        and similarity >= STRICT_SIMILARITY
    )
    if not eligible:
        return INELIGIBLE_FLOOR

    reward = (
        0.25 * float(bool(details.get("canonical")))
        + 0.50 * float(details.get("mean_satisfaction", 0.0))
        + 0.75 * float(details.get("bottleneck", 0.0))
        + 0.50 * float(details.get("property_fraction", 0.0))
    )
    if bool(details.get("property_strict")):
        reward += 1.00
    if bool(details.get("strict")):
        reward += 4.00
    return reward

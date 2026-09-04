#!/usr/bin/env python3
"""Deterministic property-program routing over the rank axis of one LoRA.

The adapter remains a standard PEFT LoRA on disk.  During training and
generation, forward hooks mask the output of every LoRA-A projection so that
only ranks addressed by the current request's property program are active.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def load_layout(path: Path) -> dict[str, object]:
    layout = json.loads(path.read_text())
    validate_layout(layout)
    return layout


def validate_layout(layout: Mapping[str, object]) -> None:
    rank = int(layout.get("rank", 0))
    if rank < 1:
        raise ValueError("routing rank must be positive")
    common = _rank_list(layout.get("common_ranks", []), rank, "common_ranks")
    if not common:
        raise ValueError("at least one common rank is required")
    property_ranks = layout.get("property_ranks")
    node_rank_weights = layout.get("node_rank_weights")
    if property_ranks is None and node_rank_weights is None:
        raise ValueError("layout requires property_ranks or node_rank_weights")
    if property_ranks is not None:
        if not isinstance(property_ranks, Mapping) or not property_ranks:
            raise ValueError("property_ranks must be a non-empty mapping")
        for prop, ranks in property_ranks.items():
            if not str(prop):
                raise ValueError("property name cannot be empty")
            if not _rank_list(ranks, rank, f"property_ranks.{prop}"):
                raise ValueError(f"property {prop} has no routed ranks")
    if node_rank_weights is not None:
        if not isinstance(node_rank_weights, Mapping) or not node_rank_weights:
            raise ValueError("node_rank_weights must be a non-empty mapping")
        for node, weights in node_rank_weights.items():
            if not str(node) or ":" not in str(node):
                raise ValueError(f"invalid mode-property routing node: {node}")
            if not isinstance(weights, Mapping) or not weights:
                raise ValueError(f"node_rank_weights.{node} must be a non-empty mapping")
            for index, value in weights.items():
                rank_index = int(index)
                weight = float(value)
                if rank_index < 0 or rank_index >= rank:
                    raise ValueError(
                        f"node_rank_weights.{node} contains rank {rank_index} outside [0, {rank})"
                    )
                if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
                    raise ValueError(
                        f"node_rank_weights.{node}.{rank_index} must be in [0, 1]"
                    )
        inactive_floor = float(layout.get("inactive_floor", 0.0))
        if not math.isfinite(inactive_floor) or not 0.0 <= inactive_floor <= 1.0:
            raise ValueError("inactive_floor must be in [0, 1]")
        if str(layout.get("combination", "max")) != "max":
            raise ValueError("transfer-aware routing currently supports combination=max")
    mode_ranks = layout.get("mode_ranks", {})
    if not isinstance(mode_ranks, Mapping):
        raise ValueError("mode_ranks must be a mapping")
    for mode, ranks in mode_ranks.items():
        _rank_list(ranks, rank, f"mode_ranks.{mode}")
    normalization = str(layout.get("normalization", "binary"))
    if normalization not in {"binary", "rms_active"}:
        raise ValueError(f"unsupported routing normalization: {normalization}")


def _rank_list(value: object, rank: int, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = [int(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate ranks")
    if any(item < 0 or item >= rank for item in result):
        raise ValueError(f"{label} contains a rank outside [0, {rank})")
    return result


def condition_program(row: Mapping[str, object]) -> list[dict[str, object]]:
    program = row.get("condition_program")
    if isinstance(program, list):
        return [dict(item) for item in program if isinstance(item, Mapping)]
    for message in list(row.get("messages", [])):
        if str(message.get("role", "")) != "user":
            continue
        payload = json.loads(str(message.get("content", "{}")))
        conditions = payload.get("conditions", []) if isinstance(payload, Mapping) else []
        if isinstance(conditions, list):
            return [dict(item) for item in conditions if isinstance(item, Mapping)]
    raise ValueError("request has no property program")


def task_mode(row: Mapping[str, object]) -> str:
    explicit = str(row.get("task_mode", ""))
    if explicit in {"de_novo", "edit"}:
        return explicit
    for message in list(row.get("messages", [])):
        if str(message.get("role", "")) != "user":
            continue
        payload = json.loads(str(message.get("content", "{}")))
        if isinstance(payload, Mapping):
            return "de_novo" if str(payload.get("source", "")) == "<EMPTY>" else "edit"
    raise ValueError("request has no recognized task mode")


def properties(row: Mapping[str, object]) -> tuple[str, ...]:
    values = tuple(str(item.get("property", "")) for item in condition_program(row))
    if not values or any(not value for value in values):
        raise ValueError("request contains an empty property name")
    return values


def route_values(row: Mapping[str, object], layout: Mapping[str, object]) -> list[float]:
    """Return one normalized rank mask for a request."""
    validate_layout(layout)
    rank = int(layout["rank"])
    common = set(int(item) for item in layout["common_ranks"])
    weights = [0.0] * rank
    for index in common:
        weights[index] = 1.0
    property_ranks = layout.get("property_ranks")
    node_rank_weights = layout.get("node_rank_weights")
    mode = task_mode(row)
    if node_rank_weights is not None:
        assert isinstance(node_rank_weights, Mapping)
        floor = float(layout.get("inactive_floor", 0.0))
        for index in range(rank):
            if index not in common:
                weights[index] = floor
        for prop in properties(row):
            node = f"{mode}:{prop}"
            if node not in node_rank_weights:
                raise ValueError(f"routing node {node} is absent from the routing layout")
            routed = node_rank_weights[node]
            assert isinstance(routed, Mapping)
            for index, value in routed.items():
                rank_index = int(index)
                weights[rank_index] = max(weights[rank_index], float(value))
    else:
        assert isinstance(property_ranks, Mapping)
        active = set(common)
        for prop in properties(row):
            if prop not in property_ranks:
                raise ValueError(f"property {prop} is absent from the routing layout")
            active.update(int(item) for item in property_ranks[prop])
        mode_ranks = layout.get("mode_ranks", {})
        assert isinstance(mode_ranks, Mapping)
        active.update(int(item) for item in mode_ranks.get(mode, []))
        for index in active:
            weights[index] = 1.0
    squared_norm = sum(value * value for value in weights)
    if squared_norm <= 0.0:
        raise ValueError("routing produced an empty rank set")
    if str(layout.get("normalization", "binary")) == "rms_active":
        scale = math.sqrt(rank / squared_norm)
        weights = [value * scale for value in weights]
    return weights


def route_matrix(rows: Sequence[Mapping[str, object]], layout: Mapping[str, object]):
    import torch

    return torch.tensor(
        [route_values(row, layout) for row in rows], dtype=torch.float32
    )


def install_lora_rank_routing(model: object, *, rank: int) -> int:
    """Install rank-mask hooks on every PEFT LoRA-A projection."""
    import torch

    installed = 0
    for layer in model.modules():
        lora_a = getattr(layer, "lora_A", None)
        if lora_a is None or not hasattr(lora_a, "items"):
            continue
        for adapter_name, projection in lora_a.items():
            if int(getattr(projection, "out_features", -1)) != rank:
                raise ValueError(
                    f"adapter {adapter_name} has rank "
                    f"{getattr(projection, 'out_features', None)}, expected {rank}"
                )
            if getattr(projection, "_molprogram_rank_routing", False):
                continue
            projection._molprogram_route_mask = torch.ones(1, rank)
            projection._molprogram_rank_routing = True

            def mask_rank_output(module, _inputs, output):
                mask = module._molprogram_route_mask.to(
                    device=output.device, dtype=output.dtype
                )
                if output.ndim < 2:
                    raise ValueError("LoRA-A output must have batch and rank axes")
                if mask.shape[0] not in {1, output.shape[0]}:
                    raise ValueError(
                        f"route batch {mask.shape[0]} does not match "
                        f"LoRA batch {output.shape[0]}"
                    )
                view = [mask.shape[0]] + [1] * (output.ndim - 2) + [rank]
                return output * mask.reshape(view)

            projection.register_forward_hook(mask_rank_output)
            installed += 1
    if installed == 0:
        raise ValueError("no PEFT LoRA-A projections were found for routing")
    return installed


def set_lora_route_mask(model: object, mask: object) -> int:
    """Set the per-example rank mask on every installed LoRA-A projection."""
    count = 0
    for module in model.modules():
        if getattr(module, "_molprogram_rank_routing", False):
            module._molprogram_route_mask = mask
            count += 1
    if count == 0:
        raise ValueError("LoRA rank routing has not been installed")
    return count

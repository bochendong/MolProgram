#!/usr/bin/env python3
"""Compile task-gradient signatures into a soft LoRA rank-routing layout."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def normalize_signatures(
    signatures: Mapping[str, Sequence[float]],
) -> tuple[list[str], np.ndarray]:
    if not signatures:
        raise ValueError("at least one gradient signature is required")
    nodes = sorted(str(node) for node in signatures)
    width = len(signatures[nodes[0]])
    if width < 1:
        raise ValueError("gradient signatures cannot be empty")
    matrix = np.asarray([signatures[node] for node in nodes], dtype=np.float64)
    if matrix.shape != (len(nodes), width):
        raise ValueError("all gradient signatures must have the same width")
    if not np.isfinite(matrix).all():
        raise ValueError("gradient signatures must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        missing = [nodes[index] for index in np.flatnonzero(norms <= 0.0)]
        raise ValueError(f"zero gradient signatures for nodes: {missing}")
    return nodes, matrix / norms[:, None]


def cosine_graph(signatures: Mapping[str, Sequence[float]]) -> dict[str, object]:
    nodes, normalized = normalize_signatures(signatures)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return {
        "nodes": nodes,
        "cosine": cosine.tolist(),
    }


def compile_signed_spectral_layout(
    signatures: Mapping[str, Sequence[float]],
    *,
    rank: int = 16,
    common_ranks: int = 8,
    inactive_floor: float = 0.25,
) -> tuple[dict[str, object], dict[str, object]]:
    """Assign residual ranks from signed eigenvectors of task-gradient affinity.

    Each graph factor receives two ranks: one for its positive loading and one
    for its negative loading. Tasks with aligned gradients therefore reuse a
    residual slot, while tasks on opposite sides of a factor are separated.
    """
    if rank < 3:
        raise ValueError("rank must be at least three")
    if common_ranks < 1 or common_ranks >= rank:
        raise ValueError("common_ranks must be in [1, rank)")
    residual = rank - common_ranks
    if residual % 2:
        raise ValueError("signed spectral routing requires an even residual rank count")
    if not 0.0 < inactive_floor < 1.0:
        raise ValueError("inactive_floor must be strictly between zero and one")

    nodes, normalized = normalize_signatures(signatures)
    affinity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(affinity)
    order = np.argsort(eigenvalues)[::-1]
    factor_count = min(residual // 2, len(nodes))
    selected_values = np.maximum(eigenvalues[order[:factor_count]], 0.0)
    selected_vectors = eigenvectors[:, order[:factor_count]]
    leading = float(selected_values[0]) if len(selected_values) else 0.0
    if leading <= 0.0:
        raise ValueError("gradient affinity has no positive spectral factor")

    node_rank_weights: dict[str, dict[str, float]] = {}
    for node_index, node in enumerate(nodes):
        weights: dict[str, float] = {}
        for factor in range(factor_count):
            vector = selected_vectors[:, factor]
            denominator = float(np.max(np.abs(vector)))
            strength = math.sqrt(float(selected_values[factor]) / leading)
            loading = float(vector[node_index]) / max(denominator, 1e-12)
            positive_rank = common_ranks + factor * 2
            negative_rank = positive_rank + 1
            weights[str(positive_rank)] = max(0.0, loading) * strength
            weights[str(negative_rank)] = max(0.0, -loading) * strength
        node_rank_weights[node] = weights

    layout = {
        "name": "transfer_aware_signed_spectral_rank16_v1",
        "rank": rank,
        "normalization": "rms_active",
        "common_ranks": list(range(common_ranks)),
        "inactive_floor": inactive_floor,
        "combination": "max",
        "node_rank_weights": node_rank_weights,
        "compiler": {
            "type": "signed_spectral_gradient_affinity",
            "factor_count": factor_count,
            "positive_negative_rank_pairs": [
                [common_ranks + 2 * factor, common_ranks + 2 * factor + 1]
                for factor in range(factor_count)
            ],
        },
    }
    evidence = {
        "nodes": nodes,
        "cosine": affinity.tolist(),
        "eigenvalues": [float(value) for value in selected_values],
        "factor_loadings": selected_vectors.tolist(),
    }
    return layout, evidence

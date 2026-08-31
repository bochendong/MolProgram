"""Target-blind molecular property scoring used by evaluation and RL."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors

from . import protocol


PROPERTY_NORMALIZERS = {
    "MW": 40.0,
    "LogP": 1.0,
    "QED": 0.1,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
    "SA": 1.0,
    "GSK3B": 0.1,
    "DRD2": 0.1,
    "JNK3": 0.1,
}
STRICT_TOLERANCE = {
    "MW": 20.0,
    "LogP": 0.5,
    "QED": 0.05,
    "TPSA": 10.0,
    "HBD": 0.5,
    "HBA": 0.5,
    "RB": 0.5,
}


@dataclass(frozen=True)
class PropertyScore:
    success_fraction: float
    mean_satisfaction: float
    bottleneck: float
    all_success: bool
    evaluated_count: int
    property_count: int


def prompt_payload(row: Mapping[str, object]) -> dict[str, object]:
    for message in list(row.get("messages", [])):
        if str(message.get("role")) == "user":
            payload = json.loads(str(message.get("content", "{}")))
            if isinstance(payload, dict):
                return payload
    program = row.get("condition_program")
    if isinstance(program, list):
        source = str(row.get("source", row.get("source_smiles", "")) or "")
        return {"conditions": program, "source": source or "<EMPTY>"}
    raise ValueError("row does not contain a structured MolProgram request")


def property_count(row: Mapping[str, object]) -> int:
    conditions = prompt_payload(row).get("conditions", [])
    return len(conditions) if isinstance(conditions, list) else 0


@lru_cache(maxsize=200_000)
def molecular_properties(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": float(Descriptors.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "RB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    }


@lru_cache(maxsize=200_000)
def score_property(smiles: str, prop: str) -> float | None:
    canonical = protocol.canonical_smiles(smiles)
    if not canonical:
        return None
    if prop in molecular_properties(canonical):
        return molecular_properties(canonical)[prop]
    try:
        from tdc import Oracle

        return float(Oracle(name=prop)(canonical))
    except Exception:
        return None


def morgan_tanimoto(left: str, right: str) -> float:
    left_mol = Chem.MolFromSmiles(left)
    right_mol = Chem.MolFromSmiles(right)
    if left_mol is None or right_mol is None:
        return 0.0
    left_fp = AllChem.GetMorganFingerprintAsBitVect(left_mol, 2, nBits=2048)
    right_fp = AllChem.GetMorganFingerprintAsBitVect(right_mol, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(left_fp, right_fp))


def _condition_margin(
    condition: Mapping[str, object], candidate: str, source: str
) -> float | None:
    prop = str(condition.get("property", ""))
    if prop not in protocol.PROPERTIES:
        return None
    value = score_property(candidate, prop)
    if value is None:
        return None
    goal = condition.get("goal")
    if isinstance(goal, Mapping) and "around" in goal:
        tolerance = STRICT_TOLERANCE.get(prop, PROPERTY_NORMALIZERS.get(prop, 1.0))
        return 1.0 - abs(value - float(goal["around"])) / max(tolerance, 1e-8)
    if isinstance(goal, str) and source and source != "<EMPTY>":
        source_value = score_property(source, prop)
        if source_value is None:
            return None
        scale = max(PROPERTY_NORMALIZERS.get(prop, 1.0), 1e-8)
        if goal == "increase":
            return (value - source_value) / scale
        if goal == "decrease":
            return (source_value - value) / scale
        if goal == "preserve":
            return 1.0 - abs(value - source_value) / scale
    return None


def score_properties(payload: Mapping[str, object], candidate: str) -> PropertyScore:
    conditions = payload.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        return PropertyScore(0.0, 0.0, 0.0, False, 0, 0)
    source = str(payload.get("source", "") or "")
    margins = [
        margin
        for condition in conditions
        if isinstance(condition, Mapping)
        for margin in [_condition_margin(condition, candidate, source)]
        if margin is not None
    ]
    property_total = len(conditions)
    successes = sum(margin >= 0.0 for margin in margins)
    satisfaction = [0.5 * (math.tanh(margin / 0.25) + 1.0) for margin in margins]
    if margins:
        minimum = min(margins)
        softmin = minimum - 0.25 * math.log(
            sum(math.exp(-(margin - minimum) / 0.25) for margin in margins)
        )
        bottleneck = 0.5 * (math.tanh(softmin / 0.25) + 1.0)
    else:
        bottleneck = 0.0
    return PropertyScore(
        success_fraction=successes / max(property_total, 1),
        mean_satisfaction=sum(satisfaction) / max(property_total, 1),
        bottleneck=bottleneck,
        all_success=len(margins) == property_total and successes == property_total,
        evaluated_count=len(margins),
        property_count=property_total,
    )


def score_response(
    row: Mapping[str, object], raw: str
) -> tuple[float, dict[str, object]]:
    """Score prompt-visible constraints only; target molecules are never read."""
    mode = str(row.get("task_mode", "")) or (
        "de_novo" if str(prompt_payload(row).get("source")) == "<EMPTY>" else "edit"
    )
    parsed = protocol.parse_response(raw, mode)
    if not parsed.get("valid"):
        return -1.0, {
            "valid": False,
            "canonical": False,
            "property_strict": False,
            "strict": False,
            "relaxed": False,
            "property_fraction": 0.0,
            "source_similarity": None,
            "copy": False,
        }
    candidate = str(parsed["smiles"])
    payload = prompt_payload(row)
    properties = score_properties(payload, candidate)
    reward = (
        0.50
        + 0.10 * float(bool(parsed.get("canonical")))
        + 0.75 * properties.success_fraction
        + 0.75 * properties.mean_satisfaction
        + 0.75 * properties.bottleneck
        + 1.00 * float(properties.all_success)
    )
    similarity: float | None = None
    copy = False
    strict = properties.all_success
    relaxed = properties.all_success
    if mode == "edit":
        source = protocol.canonical_smiles(payload.get("source", ""))
        similarity = morgan_tanimoto(source, candidate)
        copy = bool(source and source == candidate)
        relaxed = bool(properties.all_success and similarity >= 0.15)
        strict = bool(properties.all_success and similarity >= 0.65)
        reward += 0.25 * min(similarity / 0.65, 1.0)
        reward += 0.25 * float(similarity >= 0.15)
        reward += 0.50 * float(similarity >= 0.65)
        reward += 0.50 * float(relaxed)
        reward += 1.00 * float(strict)
        reward -= 0.25 * float(copy)
    return reward, {
        "valid": True,
        "canonical": bool(parsed.get("canonical")),
        "property_strict": properties.all_success,
        "strict": strict,
        "relaxed": relaxed,
        "property_fraction": properties.success_fraction,
        "mean_satisfaction": properties.mean_satisfaction,
        "bottleneck": properties.bottleneck,
        "source_similarity": similarity,
        "copy": copy,
    }

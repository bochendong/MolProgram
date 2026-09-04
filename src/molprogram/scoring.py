"""Target-blind molecular property scoring used by evaluation and RL."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, RDConfig, rdMolDescriptors

from . import protocol


PROPERTY_NORMALIZERS = {
    "MW": 500.0,
    "LogP": 6.0,
    "QED": 1.0,
    "TPSA": 160.0,
    "HBD": 8.0,
    "HBA": 12.0,
    "RB": 12.0,
    "SA": 8.0,
    "GSK3B": 0.5,
    "DRD2": 0.5,
    "JNK3": 0.5,
}
STRICT_TOLERANCE = {
    "MW": 35.0,
    "LogP": 1.0,
    "QED": 0.10,
    "TPSA": 20.0,
    "HBD": 1.0,
    "HBA": 1.0,
    "RB": 1.0,
}
PINNED_ORACLE_ENVS = {
    "GSK3B": "SUCC_GSK3B_ORACLE_PATH",
    "DRD2": "SUCC_DRD2_ORACLE_PATH",
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
        "LogP": float(Crippen.MolLogP(mol)),
        "QED": float(QED.qed(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
        "RB": float(Lipinski.NumRotatableBonds(mol)),
    }


class PinnedMorganClassifierOracle:
    """Apply the frozen benchmark assay model to its original fingerprint."""

    def __init__(self, model_path: Path, prop: str):
        self.model_path = model_path.resolve()
        self.prop = str(prop).upper()
        with self.model_path.open("rb") as handle, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.prop == "DRD2":
                import sklearn.svm._classes as svm_classes

                sys.modules.setdefault("sklearn.svm.classes", svm_classes)
            self.model = pickle.load(handle)
        if self.prop == "DRD2":
            old = vars(self.model)
            for name, value in {
                "_n_support": old.get("n_support_"),
                "_probA": old.get("probA_"),
                "_probB": old.get("probB_"),
            }.items():
                if not hasattr(self.model, name) and value is not None:
                    setattr(self.model, name, value)
            if not hasattr(self.model, "n_features_in_"):
                shape_fit = getattr(self.model, "shape_fit_", None)
                if not shape_fit or len(shape_fit) != 2:
                    raise ValueError("Pinned DRD2 SVC is missing shape_fit_")
                self.model.n_features_in_ = int(shape_fit[1])
            if not hasattr(self.model, "break_ties"):
                self.model.break_ties = False

    def __call__(self, smiles: str) -> float:
        molecule = Chem.MolFromSmiles(str(smiles or ""))
        if molecule is None:
            raise ValueError("invalid SMILES")
        features = np.zeros(2048, dtype=np.float32)
        if self.prop == "DRD2":
            fingerprint = AllChem.GetMorganFingerprint(
                molecule, 3, useCounts=True, useFeatures=True
            )
            for index, count in fingerprint.GetNonzeroElements().items():
                features[int(index) % 2048] += float(count)
        else:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, 2, nBits=2048
            )
            DataStructs.ConvertToNumpyArray(fingerprint, features)
        probability = self.model.predict_proba(features.reshape(1, -1))
        return float(probability[0, 1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_oracle_provenance() -> dict[str, dict[str, str]]:
    provenance = {}
    for prop, env_name in PINNED_ORACLE_ENVS.items():
        configured = str(os.environ.get(env_name, "") or "").strip()
        if not configured:
            continue
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        provenance[prop] = {
            "implementation": "pinned_benchmark_morgan_sklearn_classifier",
            "env": env_name,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    return provenance


@lru_cache(maxsize=None)
def _pinned_oracle(prop: str, configured_path: str) -> PinnedMorganClassifierOracle:
    path = Path(configured_path).expanduser().resolve()
    if not path.is_file():
        env_name = PINNED_ORACLE_ENVS[prop]
        raise FileNotFoundError(f"{env_name} does not exist: {path}")
    return PinnedMorganClassifierOracle(path, prop)


@lru_cache(maxsize=1)
def _sa_oracle():
    scorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    if not scorer_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("molprogram_sascorer", scorer_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def score(smiles: str) -> float:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("invalid SMILES")
        return float(module.calculateScore(molecule))

    return score


@lru_cache(maxsize=None)
def _tdc_oracle(prop: str):
    try:
        from tdc import Oracle

        return Oracle(name=prop)
    except Exception:
        return None


@lru_cache(maxsize=200_000)
def _score_property(canonical: str, prop: str, pinned_path: str) -> float | None:
    properties = molecular_properties(canonical)
    if prop in properties:
        return properties[prop]
    if prop == "SA":
        oracle = _sa_oracle()
    elif pinned_path:
        oracle = _pinned_oracle(prop, pinned_path)
    else:
        oracle = _tdc_oracle(prop)
    if oracle is None:
        return None
    try:
        value = float(oracle(canonical))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def score_property(smiles: str, prop: str) -> float | None:
    canonical = protocol.canonical_smiles(smiles)
    canonical_prop = str(prop or "").strip()
    if not canonical or not canonical_prop:
        return None
    env_name = PINNED_ORACLE_ENVS.get(canonical_prop)
    pinned_path = str(os.environ.get(env_name, "") or "").strip() if env_name else ""
    return _score_property(canonical, canonical_prop, pinned_path)


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

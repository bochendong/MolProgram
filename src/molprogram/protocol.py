#!/usr/bin/env python3
"""Target-blind prompt and response contract for MolProgram."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Mapping

from rdkit import Chem


PROTOCOL = "molprogram_v1"
PROPERTIES = (
    "MW", "LogP", "QED", "TPSA", "HBD", "HBA", "RB", "SA",
    "GSK3B", "DRD2", "JNK3",
)
PROPERTY_ORDER = {name: index for index, name in enumerate(PROPERTIES)}
TABLE1_TASK_KEYS = {
    frozenset({("GSK3B", "increase")}): "GSK3B:increase",
    frozenset({("RB", "decrease")}): "RB:decrease",
    frozenset({("MW", "increase")}): "MW:increase",
    frozenset({("SA", "decrease")}): "SA:decrease",
    frozenset({("HBA", "decrease"), ("SA", "decrease")}): "HBA:decrease+SA:decrease",
    frozenset({("QED", "increase"), ("SA", "decrease")}): "QED:increase+SA:decrease",
    frozenset({("HBA", "decrease"), ("LogP", "increase")}): "HBA:decrease+LogP:increase",
    frozenset({("HBA", "decrease"), ("MW", "decrease")}): "HBA:decrease+MW:decrease",
    frozenset({("DRD2", "decrease"), ("MW", "decrease"), ("SA", "decrease")}): (
        "DRD2:decrease+MW:decrease+SA:decrease"
    ),
    frozenset({("HBA", "increase"), ("MW", "increase"), ("QED", "decrease")}): (
        "HBA:increase+MW:increase+QED:decrease"
    ),
}
SYSTEM = (
    "You are one molecular causal language model for de-novo construction and "
    "source-conditioned editing. Follow every explicitly listed condition. Infer "
    "the mode only from whether source is <EMPTY> or a molecule. Return exactly "
    '{"plan":"BUILD","smiles":"CANONICAL_SMILES"} or '
    '{"plan":"MODIFY","smiles":"CANONICAL_SMILES"}. No prose or markdown.'
)
RESPONSE_RE = re.compile(r'^\{"plan":"(BUILD|MODIFY)","smiles":"([^"\n]+)"\}$')


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return re.sub(r"[\s_\-]+", "", text)


PROPERTY_ALIASES = {
    "mw": "MW", "molwt": "MW", "molecularweight": "MW",
    "logp": "LogP", "plogp": "LogP", "qed": "QED", "tpsa": "TPSA",
    "hbd": "HBD", "hbonddonor": "HBD", "hba": "HBA",
    "haccept": "HBA", "hbondacceptor": "HBA", "rb": "RB",
    "rotbond": "RB", "rotbonds": "RB", "rotatablebonds": "RB",
    "sa": "SA", "sas": "SA", "syntheticaccessibility": "SA",
    "gsk3b": "GSK3B", "gsk3β": "GSK3B", "gsk3beta": "GSK3B",
    "gsk3": "GSK3B", "drd2": "DRD2", "jnk3": "JNK3",
}
DIRECTION_ALIASES = {
    "increase": "increase", "increased": "increase", "up": "increase",
    "higher": "increase", "raise": "increase", "boost": "increase", "+": "increase",
    "↑": "increase", "decrease": "decrease", "decreased": "decrease",
    "down": "decrease", "lower": "decrease", "reduce": "decrease", "-": "decrease",
    "↓": "decrease", "preserve": "preserve", "maintain": "preserve",
    "same": "preserve", "unchanged": "preserve",
}


def canonical_property(value: object) -> str:
    return PROPERTY_ALIASES.get(_normalized(value), "")


def canonical_direction(value: object) -> str:
    return DIRECTION_ALIASES.get(_normalized(value), "")


def canonical_smiles(value: object) -> str:
    text = str(value or "").strip()
    mol = Chem.MolFromSmiles(text) if text else None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else ""


def _json_value(value: object, empty: object) -> object:
    if isinstance(value, (list, dict)):
        return value
    text = str(value or "").strip()
    if not text:
        return empty
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed explicit instruction metadata") from exc


def _deduplicate(items: list[dict[str, object]]) -> list[dict[str, object]]:
    by_property: dict[str, object] = {}
    for item in items:
        prop, goal = str(item["property"]), item["goal"]
        if prop in by_property and by_property[prop] != goal:
            raise ValueError(f"conflicting explicit goals for {prop}")
        by_property[prop] = goal
    return [
        {"property": prop, "goal": by_property[prop]}
        for prop in sorted(by_property, key=lambda name: PROPERTY_ORDER[name])
    ]


def explicit_instruction_conditions(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Read only declared instruction tasks, never target-derived active deltas."""
    raw_tasks = row.get("instruction_tasks", "")
    parsed = _json_value(raw_tasks, []) if str(raw_tasks or "").strip() else []
    items: list[dict[str, object]] = []
    if parsed:
        if not isinstance(parsed, list):
            raise ValueError("instruction_tasks must be a JSON list")
        for task in parsed:
            if not isinstance(task, Mapping):
                raise ValueError("instruction task must be an object")
            prop = canonical_property(task.get("property", ""))
            direction = canonical_direction(task.get("direction", task.get("goal", "")))
            if not prop or not direction:
                raise ValueError(f"unsupported explicit task: {task}")
            items.append({"property": prop, "goal": direction})
    else:
        raw_properties = str(row.get("instruction_task_properties", "") or "").replace(",", "|")
        directions = _json_value(row.get("instruction_task_directions", ""), {})
        if raw_properties and not isinstance(directions, Mapping):
            raise ValueError("instruction_task_directions must be a JSON object")
        for raw_prop in (part.strip() for part in raw_properties.split("|") if part.strip()):
            prop = canonical_property(raw_prop)
            raw_direction = directions.get(raw_prop, directions.get(prop, "")) if isinstance(directions, Mapping) else ""
            direction = canonical_direction(raw_direction)
            if not prop or not direction:
                raise ValueError(f"unsupported explicit task property: {raw_prop}")
            items.append({"property": prop, "goal": direction})
    if not items:
        raise ValueError("edit row has no explicit instruction tasks")
    return _deduplicate(items)


def legacy_denovo_conditions(row: Mapping[str, object]) -> list[dict[str, object]]:
    """Read the declared de-novo control columns used by the balanced 2p-7p set."""
    items: list[dict[str, object]] = []
    for prop in PROPERTIES:
        active = str(row.get(f"{prop}_active", "")).strip().lower()
        none_flag = str(row.get(f"{prop}_None", "")).strip().lower()
        if active not in {"1", "true", "yes"} and none_flag != "false":
            continue
        direction = canonical_direction(row.get(f"{prop}_direction", ""))
        if direction:
            items.append({"property": prop, "goal": direction})
            continue
        raw = str(row.get(f"{prop}_setting", "") or row.get(f"target_{prop}", "")).strip()
        try:
            items.append({"property": prop, "goal": {"around": round(float(raw), 6)}})
        except ValueError as exc:
            raise ValueError(f"active de-novo property {prop} lacks a declared goal") from exc
    if not items:
        raise ValueError("de-novo row has no declared conditions")
    return _deduplicate(items)


def condition_program(row: Mapping[str, object], mode: str) -> list[dict[str, object]]:
    return explicit_instruction_conditions(row) if mode == "edit" else legacy_denovo_conditions(row)


def mode_for_source(source: str) -> str:
    return "de_novo" if source == "<EMPTY>" else "edit"


def build_prompt(row: Mapping[str, object]) -> tuple[list[dict[str, str]], str, str]:
    source_raw = str(
        row.get("source_canonical_smiles", "") or row.get("source_smiles", "") or ""
    ).strip()
    source = canonical_smiles(source_raw) if source_raw else "<EMPTY>"
    if source_raw and not source:
        raise ValueError("invalid source SMILES")
    mode = mode_for_source(source)
    payload = {"conditions": condition_program(row, mode), "source": source}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"))},
    ]
    serialized = json.dumps(messages, sort_keys=True)
    for forbidden in ("target_smiles", "target_canonical_smiles", "policy_target_smiles", "oracle"):
        if forbidden in serialized:
            raise AssertionError(f"forbidden prompt field: {forbidden}")
    return messages, source, mode


def response(target: str, mode: str) -> str:
    canonical = canonical_smiles(target)
    if not canonical:
        raise ValueError("invalid target SMILES")
    plan = "BUILD" if mode == "de_novo" else "MODIFY"
    return json.dumps({"plan": plan, "smiles": canonical}, separators=(",", ":"))


def parse_response(text: str, expected_mode: str) -> dict[str, object]:
    match = RESPONSE_RE.fullmatch(str(text).strip())
    if match is None:
        return {"strict_parse": False, "valid": False, "canonical": False, "smiles": ""}
    plan, smiles = match.groups()
    expected_plan = "BUILD" if expected_mode == "de_novo" else "MODIFY"
    canonical = canonical_smiles(smiles)
    strict = plan == expected_plan
    return {
        "strict_parse": strict, "valid": strict and bool(canonical),
        "canonical": strict and bool(canonical) and canonical == smiles,
        "smiles": canonical if strict and canonical else "", "plan": plan,
    }


def condition_hash_from_program(program: list[dict[str, object]]) -> str:
    payload = json.dumps(program, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def task_key(program: list[dict[str, object]]) -> str:
    directional = frozenset(
        (str(item["property"]), str(item["goal"]))
        for item in program if isinstance(item["goal"], str)
    )
    if len(directional) == len(program) and directional in TABLE1_TASK_KEYS:
        return TABLE1_TASK_KEYS[directional]
    parts = []
    for item in program:
        goal = item["goal"]
        label = str(goal) if isinstance(goal, str) else f"around={goal['around']}"
        parts.append(f"{item['property']}:{label}")
    return "+".join(parts)


def smiles_hash(smiles: str) -> str:
    return hashlib.sha256(smiles.encode()).hexdigest() if smiles and smiles != "<EMPTY>" else ""

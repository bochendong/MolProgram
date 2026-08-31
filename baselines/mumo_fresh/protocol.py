"""MuMO/GeLLM-compatible prompt and response contract."""

from __future__ import annotations

import re


PROPERTY_FULL_NAMES = {
    "plogp": "Penalized octanol-water partition coefficient (penalized logP)",
    "qed": "QED",
    "drd2": "DRD2 inhibition",
    "bbbp": "BBB permeability",
    "mutagenicity": "Mutagenicity",
    "hia": "Intestinal adsorption",
}

INSTRUCTIONS = (
    "Modify the given molecule to adjust the specified molecular properties by "
    "substituting functional groups while keeping changes to the core structure "
    "minimal. Output only the SMILES of the modified molecule, wrapped in "
    "<SMILES> </SMILES> tags.",
    "Your goal is to fine-tune the specified molecular properties of the given "
    "compound with minimal structural changes. Make the necessary adjustments "
    "and return the modified molecule in a SMILES format enclosed in "
    "<SMILES> </SMILES> tags.",
    "Adjust the structure of the given molecule to target the specified "
    "adjustments in molecular properties. Retain the core structure as much as "
    "possible. Respond with only the SMILES of the modified molecule enclosed "
    "in <SMILES> </SMILES> tags.",
    "Modify the given molecular structure to target specific property changes, "
    "aiming to keep structural adjustments minimal. Respond solely with the "
    "SMILES notation for the adjusted molecule, enclosed within "
    "<SMILES> </SMILES> tags.",
    "Alter the given molecule to meet the desired property changes with the "
    "least structural alteration possible. Output only the adjusted molecule in "
    "SMILES format, using <SMILES> </SMILES> tags.",
    "Your task is to modify the given molecule to adjust specific molecular "
    "properties while keeping structural changes as minimal as possible. Your "
    "response should only contain a valid SMILES representation of the modified "
    "molecule enclosed with <SMILES> </SMILES> tag.",
)


def adjustment_text(task: str) -> str:
    properties = [item for item in task.split("+") if item]
    clauses = [
        f"{'decrease' if prop == 'mutagenicity' else 'increase'} "
        f"{PROPERTY_FULL_NAMES[prop]}"
        for prop in properties
    ]
    if not clauses:
        raise ValueError(f"empty MuMO task: {task!r}")
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]


def user_text(source_smiles: str, task: str, instruction_index: int) -> str:
    if instruction_index < 0 or instruction_index >= len(INSTRUCTIONS):
        raise ValueError(f"unsupported instruction index: {instruction_index}")
    return (
        f"{INSTRUCTIONS[instruction_index]}\n"
        f"%%% Input : <SMILES> {source_smiles} </SMILES>\n"
        f"%%% Adjust: {adjustment_text(task)}\n"
    )


def messages(
    source_smiles: str,
    task: str,
    instruction_index: int,
    target_smiles: str | None = None,
):
    rows = [
        {
            "role": "user",
            "content": user_text(source_smiles, task, instruction_index),
        }
    ]
    if target_smiles is not None:
        rows.append(
            {
                "role": "assistant",
                "content": f"%%% Response: <SMILES> {target_smiles} </SMILES>",
            }
        )
    return rows


def extract_smiles(text: str) -> str:
    matches = re.findall(
        r"<SMILES>\s*(.*?)\s*</SMILES>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return matches[-1].strip()
    cleaned = text.split("%%% Response:")[-1].strip()
    return cleaned.splitlines()[0].strip() if cleaned else ""

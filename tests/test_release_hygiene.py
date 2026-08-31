import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_has_no_internal_experiment_paths():
    forbidden = re.compile(
        r"(?:/scratch/|/home/bdong|SketchMol-Understanding-Condition|"
        r"unified_constraint_agent|unified_smiles_generator|\b[Pp]\d{1,2}(?:[._-]|\b))"
    )
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.parts):
            continue
        if path.name == Path(__file__).name:
            continue
        if path.suffix not in {".py", ".md", ".sh", ".toml", ".yml", ".yaml"}:
            continue
        if forbidden.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

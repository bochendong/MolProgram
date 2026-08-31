# Ablations

This directory is the single entry point for MolProgram ablation studies. Each
study keeps its frozen protocol, executable code, output contract, and
interpretation boundary together. Main training and reinforcement-learning
recipes remain outside this directory.

## Study index

| Study | Question | Status | Primary comparison |
|---|---|---|---|
| [Joint versus specialists](joint_vs_specialists/) | Does sharing one policy provide positive transfer or parameter efficiency? | Runnable | One joint adapter versus two task-specific adapters |
| [Negative refinement](negative_refinement/) | Do constructed negative completions improve the policy beyond matched positive-only training? | Protocol ready | Positive-only versus multi-negative contrastive refinement |

## Reporting rules

- Freeze the data subset, evaluation prompts, decoding seed, and decision rule
  before GPU training.
- Change only the component named by the ablation.
- Evaluate both de novo construction and source-conditioned editing, even when
  the intervention is expected to help only one mode.
- Report Raw@1, validity, strict success, and the per-mode result. Do not hide
  cross-task interference inside a single average.
- Preserve negative and null results and report every preregistered data scale.
- Keep target molecules unavailable to generation, candidate selection, and
  reinforcement-learning rewards.

Historical internal runs have been normalized into the protocol files under
each study. Cluster job identifiers and machine-specific paths are deliberately
excluded from the public release.

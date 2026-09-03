# Ablations

This directory is the single entry point for MolProgram ablation studies. Each
study keeps its frozen protocol, executable code, output contract, and
interpretation boundary together. Main training and reinforcement-learning
recipes remain outside this directory.

## Study index

| Study | Question | Status | Primary comparison |
|---|---|---|---|
| [Fresh balanced](fresh_balanced/) | How do exposure and task-aligned refresh affect a balanced adapter trained from the unadapted backbone? | Runnable | 100k, 200k, 500k, and full exposure; final before versus after refresh |
| [Joint versus specialists](joint_vs_specialists/) | Does sharing one policy provide positive transfer or parameter efficiency? | Runnable | One joint adapter versus two task-specific adapters |
| [Negative refinement](negative_refinement/) | Do constructed negative completions improve the policy beyond matched positive-only training? | Protocol ready | Positive-only versus multi-negative contrastive refinement |
| [Property-program routing](property_program_routing/) | Can prompt-addressed LoRA rank sharing retain Shared-5 transfer while reducing Edit-only-5 interference? | 10k pilot preregistered | Routed rank-16 versus the byte-identical fresh 10k joint baseline |
| [Safe joint Raw@1 GRPO](safe_joint_raw1_grpo/) | Can paired shared-policy RL improve de novo Raw@1 without materially degrading editing? | Protocol and Slurm DAG ready | GRPO versus the fresh input policy and matched continued SFT |

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

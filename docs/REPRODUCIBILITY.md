# Reproducibility contract

## Shared-policy comparison

The joint and specialist arms must share:

- the same unadapted base checkpoint;
- the same LoRA rank and target modules;
- the same number of examples per task;
- the same optimizer, precision, and number of epochs;
- the same target-blind prompts and sampling seeds.

The joint arm sees construction plus editing examples and produces one adapter.
Each specialist sees only its task's matched subset, producing two adapters in
total. Report both task scores and adapter parameter count; averaging the two
tasks alone can hide negative transfer.

## De novo evaluation

- Report Raw@1 on frozen 2p-7p property programs.
- If reporting Best-of-K, state K and the property-aware finalizer explicitly.
- Never compare Raw@1 against a baseline's Best-of-K without labeling the
  mismatch.

## Editing evaluation

- Report strict property success jointly with source similarity.
- The default strict source boundary is Morgan Tanimoto >= 0.65.
- Also report relaxed success, validity, source-copy rate, and mean similarity.
- Candidate generation and selection must not access the target molecule.

## Reinforcement learning

Before editing RL, run `scripts/audit_editing_reward_support.py` on training
prompts excluded from every evaluation gate. The audit reports candidate-level
source feasibility, strict Any@K, mixed-strict groups, hard-reward variance,
reward-ranking accuracy, and task coverage. Its raw group file is append-only
so an interrupted oracle-heavy audit can resume without resampling completed
prompts. A low-support decision blocks online editing RL and motivates
structured-action data collection or verified-success distillation instead.
The frozen sampling and decision contract is recorded in
`audits/editing_reward_support/preregistration.json`.

`scripts/train_rl.py` samples groups from the current policy, computes rewards
from prompt-visible conditions and the optional source, and applies
group-relative policy gradients. An SFT loss anchors the chosen completion and
an initial-adapter penalty constrains drift. Run metadata records whether any
target molecule was available to the reward function; the required value is
`false`.

The hard-boundary editing reward in `molprogram.rewards` gives every invalid,
copying, or source-infeasible output the same floor. Property rewards can exceed
that floor only after validity, non-copy, and the source-similarity constraint
are satisfied.

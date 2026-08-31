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

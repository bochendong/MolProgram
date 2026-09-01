# Editing reward-support audit

This target-blind audit decides whether the current direct-SMILES editing
policy has enough on-policy support for a small online-RL pilot. It is not a
training method and cannot establish an RL improvement.

The frozen audit samples 50 training prompts from each of the ten MolEdit
editing tasks and generates 32 candidates per prompt. It measures whether
strict, source-feasible candidates occur in mixed groups and whether the
hard-boundary reward ranks those candidates above failures. Target molecules
are unavailable to both generation and scoring.

Run from the repository root:

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export INPUT_ADAPTER=/path/to/molprogram-sft-adapter
export TRAIN_JSONL=/path/to/frozen/rl-train.jsonl
bash recipes/audit_editing_support.sh
```

On a Slurm cluster, set the same three paths and submit with:

```bash
export AUDIT_BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export AUDIT_ADAPTER_DIR=/path/to/molprogram-sft-adapter
export AUDIT_TRAIN_JSONL=/path/to/frozen/rl-train.jsonl
export AUDIT_LABEL=joint
bash audits/editing_reward_support/submit.sh
```

The append-only `groups.live.jsonl` supports restart after interruption. The
final `support_report.json` records one of four decisions:

- `DO_NOT_RUN_ONLINE_RL_SUPPORT_TOO_LOW`: change the action space or distill
  verified successes before RL;
- `REPAIR_REWARD_BEFORE_ONLINE_RL`: correct reward ranking and rerun the audit;
- `BUILD_SUPPORT_BEFORE_ONLINE_RL`: improve task coverage before RL;
- `PROCEED_TO_SMALL_ONLINE_RL_PILOT`: authorize only a small, matched pilot.

The thresholds and decision policy are frozen in `preregistration.json`.

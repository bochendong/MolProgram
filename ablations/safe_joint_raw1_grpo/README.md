# Safe joint Raw@1 GRPO

This method gate asks whether one shared MolProgram policy can improve de novo
Raw@1 without materially degrading source-conditioned editing. It starts from
the final **fresh, balanced, non-task-aligned** SFT adapter. It never starts
from a historical inherited adapter or the task-aligned refresh.

Every optimizer step pairs one de novo prompt with one editing prompt. The 30
steps cover all six de novo arities and all ten registered editing tasks with
equal within-mode exposure. Each prompt uses 16 training rollouts. Rewards are
candidate-level and prompt-visible; there is no soft OR across the group, no
best-of-K objective, and no target-molecule access. Evaluation always generates
exactly one candidate per request and performs no reranking.

The policy uses per-channel group-relative advantages, a real token-level KL
against the frozen input adapter, per-mode SFT anchors, and an equal-norm
bisector of separately differentiated de novo and editing losses. Checkpoints
10, 20, and 30 are selected on a frozen development gate. A matched
continued-SFT control receives the same paired prompts, optimizer steps,
gradient merge, and checkpoint-selection budget, but not the rollout compute.

Online RL is blocked unless the input policy passes the preregistered editing
reward-support audit. The held-out confirmation requires RL to beat both the
initial fresh SFT policy and continued SFT under the locked non-inferiority
margins in [`preregistration.json`](preregistration.json).

Run locally on one BF16 GPU after preparing the frozen inputs:

```bash
export MODEL=/path/to/base-model
export INPUT_ADAPTER=/path/to/fresh-balanced-final/adapter
export TRAIN_JSONL=/path/to/frozen/rl-train.jsonl
export SUPPORT_REPORT=/path/to/support_report.json
bash ablations/safe_joint_raw1_grpo/run_train.sh
```

For Slurm, set the `SAFE_GRPO_*` paths documented at the top of
[`submit_slurm.sh`](submit_slurm.sh). An optional `SAFE_GRPO_DEPENDENCY` can point to
the fresh-balanced training job. The submission keeps development selection
and final evaluation in separate dependent jobs.

If the preregistered development and final gates are stored as combined-mode
JSONL files, run `prepare_frozen_inputs.py` once to create the four immutable
mode-specific paths and a SHA-256 manifest consumed by the Slurm submission.

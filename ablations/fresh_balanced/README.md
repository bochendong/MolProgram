# Fresh balanced training

This experiment trains the headline balanced schedule from the unadapted base
model with a new rank-16 LoRA. It never loads an inherited adapter.

The schedule gives equal exposure to six de novo arity buckets and seven
editing arity buckets. With a physical batch of 1 and gradient accumulation of
65, each optimizer step still consumes 5 examples from every bucket (65
examples in total). Evaluation-only adapters are retained at four cumulative
exposures:

| optimizer step | total examples | de novo | editing | per bucket |
| ---: | ---: | ---: | ---: | ---: |
| 1,539 | 100,035 | 46,170 | 53,865 | 7,695 |
| 3,077 | 200,005 | 92,310 | 107,695 | 15,385 |
| 7,693 | 500,045 | 230,790 | 269,255 | 38,465 |
| 16,283 | 1,058,395 | 488,490 | 569,905 | 81,415 |

These values are cumulative balanced-stream exposure, not examples per mode.
Regular rotating checkpoints keep optimizer state for interruption recovery;
the milestone directories keep adapter weights for frozen Raw@1 evaluation.

## Stability correction

The original v1 smoke job (`20967745`) became non-finite before step 20. The
post-failure v2 protocol makes the same correction already validated by the
fresh MuMO baseline: it changes variable-length training from physical batch 5
to physical batch 1, raises accumulation from 13 to 65 so the effective batch
and per-bucket exposure are unchanged, and lowers the learning rate from
`8e-5` to `2e-5`. The smoke gate checks every microbatch loss and gradient.
Full training checks gradients before every optimizer update and adapter values
after every update, so a bad update cannot silently poison later checkpoints.
The corrected run writes to `outputs/fresh_balanced/stable_v2_seed_36001`; the
old `seed_36001/smoke/checkpoint-20` is retained only as failure evidence and is
never used for resume.
At the measured smoke throughput, the full run is expected to take roughly 59
hours, so the Slurm chain requests the partition maximum of 72 hours.

Run a smoke test and then the full fresh training:

```bash
MODEL=/path/to/base-model DATA_ROOT=/path/to/release \
  RUN_MODE=smoke bash ablations/fresh_balanced/run_train.sh

MODEL=/path/to/base-model DATA_ROOT=/path/to/release \
  RUN_MODE=full bash ablations/fresh_balanced/run_train.sh
```

The controlled comparisons are the exposure curve and the final checkpoint
before versus after the frozen task-aligned refresh. A legacy inherited-adapter
run may be shown as historical context, but it is not an initialization-only
ablation because its optimization recipe also differs.

On Slurm, `submit_slurm.sh` creates a preflight, smoke, and fresh full-training
dependency chain. Set `FRESH_BEGIN` when the cluster must defer the chain until
after a maintenance window. The command prints the final adapter and completion
marker paths needed by downstream evaluations and safe joint RL.

## Frozen Raw@1 evaluation

The headline gate lives in `benchmarks/fresh_balanced_raw1/`. It contains 440
target-blind de novo requests (100 each at 2p--5p and 20 each at 6p--7p) and
5,000 target-blind editing requests (500 for each registered All-10 task). Its
manifest freezes both output hashes and the hashes of the four source prompt
files; no assistant completion or target molecule is included.

`evaluation_protocol.json` preregisters one shared sampling seed and decision
rule for the 100k, 200k, 500k, and full adapters. The three milestones are an
exposure curve only. Only the full adapter can replace the historical balanced
checkpoint, and it must reach both 53.45% pooled de novo strict success and
56.94% All-10 editing strict success (within 2 percentage points of the frozen
historical references). The collector reports validity, property success, and
editing source similarity for All-10, Shared-5, and Edit-only-5, together with
gate, checkpoint, candidate, summary, and final-result hashes.

Submit the automatic evaluation chain after the fresh training job is known:

```bash
FRESH_EVAL_MODEL=/path/to/base-model \
FRESH_EVAL_TRAIN_ROOT=/path/to/stable_v2_seed_36001 \
FRESH_EVAL_GATE_DIR="$PWD/benchmarks/fresh_balanced_raw1" \
FRESH_EVAL_TRAIN_JOB=123456 \
FRESH_EVAL_SAFE_GRPO_JOB=123457 \
  bash ablations/fresh_balanced/submit_evaluation_slurm.sh
```

All four GPU evaluations depend on successful fresh training. The CPU collector
then verifies exact request identity and all hashes before applying the frozen
rule. If `FRESH_EVAL_SAFE_GRPO_JOB` is supplied, that held job is released only
when the full checkpoint passes both thresholds and every integrity check.

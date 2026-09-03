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

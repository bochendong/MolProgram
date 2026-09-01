# Fresh balanced training

This experiment trains the headline balanced schedule from the unadapted base
model with a new rank-16 LoRA. It never loads an inherited adapter.

The schedule gives equal exposure to six de novo arity buckets and seven
editing arity buckets. With a physical batch of 5 and gradient accumulation of
13, each optimizer step consumes 5 examples from every bucket (65 examples in
total). Evaluation-only adapters are retained at four cumulative exposures:

| optimizer step | total examples | de novo | editing | per bucket |
| ---: | ---: | ---: | ---: | ---: |
| 1,539 | 100,035 | 46,170 | 53,865 | 7,695 |
| 3,077 | 200,005 | 92,310 | 107,695 | 15,385 |
| 7,693 | 500,045 | 230,790 | 269,255 | 38,465 |
| 16,283 | 1,058,395 | 488,490 | 569,905 | 81,415 |

These values are cumulative balanced-stream exposure, not examples per mode.
Regular rotating checkpoints keep optimizer state for interruption recovery;
the milestone directories keep adapter weights for frozen Raw@1 evaluation.

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

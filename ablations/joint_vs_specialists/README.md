# Joint versus specialists

This matched ablation asks whether one shared MolProgram adapter is preferable
to a routed system with separate construction and editing adapters.

Three fresh rank-16 LoRA arms start from the same unadapted backbone:

- `joint`: the same number of de novo and editing examples;
- `denovo`: exactly the de novo subset used by `joint`;
- `edit`: exactly the editing subset used by `joint`.

The comparison holds the backbone, LoRA configuration, optimizer, epochs,
examples per task, evaluation prompts, and decoding seeds fixed. The joint arm
is compared with each specialist on its own task. The collector also reports
the parameter ratio of one adapter versus two.

## Run

From the repository root:

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export SOURCE=data/molprogram
export PROMPTS=data/eval/denovo.jsonl
export EDIT_PROMPTS=data/eval/edit.jsonl
export TRAIN_PER_TASK=10000
bash ablations/joint_vs_specialists/run.sh
```

The default output root is `outputs/joint-vs-specialists`. It contains frozen
training subsets and gates under `data/`, one model and Raw@1 evaluation per
arm, and the final comparison at `result/result.json`.

## Decision rule

The shared policy is non-inferior only when its strict success and validity are
within two percentage points of both specialists. Positive transfer additionally
requires a gain of at least two percentage points on one task. If it is merely
non-inferior, the result supports parameter efficiency, not positive transfer.

## Historical protocols

The `protocols/` directory records the sequence of earlier controlled studies:

- a large shared-initialization continuation comparison;
- fresh-adapter pilots at 3,000 and 10,000 examples per task;
- a nested scale sweep at 3,000, 10,000, 30,000, and 100,000 examples per task.

The current public runner implements the clean fresh-adapter comparison and can
reproduce any of those fresh-adapter scales by setting `TRAIN_PER_TASK`.

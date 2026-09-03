# MuMO fresh-LoRA baseline

This baseline measures what an editing-only model learns from MuMOInstruct
without inheriting any MolProgram adapter. It starts from the same unadapted
Qwen backbone with a fresh rank-16 LoRA and retains the original MuMO/GeLLM
prompt and `<SMILES>...</SMILES>` response contract.

The indexed training release contains 228,076 rows across all 63 non-empty
combinations of six MuMO properties, capped at 10,000 rows per task.

## Stability correction

An earlier internal run became non-finite during its first few optimizer steps
but continued for a full epoch because it checked parameters only at the end.
The corrected baseline changes the numerical contract before rerunning:

- excludes `lm_head` from LoRA targets and uses the same seven projection
  modules as the stable MolProgram fresh-LoRA comparisons;
- reduces the learning rate from `1e-4` to `2e-5`;
- uses physical batch 1 with accumulation 128, preserving effective batch 128;
  this follows the previously validated fresh-LoRA path and avoids the
  variable-length multi-example batches implicated by the first repaired
  smoke test;
- checks every microbatch loss, every backward pass in the first effective
  batch, every optimizer-step gradient, and every optimizer-step trainable
  parameter for finite values; the first-batch guard reports exact source rows;
- disables NaN/Inf log filtering so numerical failures cannot appear as zero
  loss;
- runs a fresh 50-step smoke test before the full one-epoch job is released.

The smoke model is a safety gate only. The full arm starts again from the clean
base model and never resumes from the smoke or failed adapter.

## Official Raw@1 evaluation

`submit_raw1.sh` evaluates the validated final adapter on the frozen ten-task
MuMO gate: 1,992 target-free conditions, one sampled completion per condition,
and no property reranking or validity repair. Each gate row is joined back to
the original test JSON so generation uses its recorded instruction variant.
The dependent scoring job uses the same ADMET-AI/TDC oracle pipeline as the
repository's existing MuMO evaluation and reports official SR, similarity and
relative improvement, plus validity and source-preserving strict success.

Raw@1 is reported separately from literature numbers using 20 candidates per
input; the two budgets must not be presented as a like-for-like comparison.

## Submit

Prepare the indexed JSONL with `prepare_data.py`, or point to a frozen release
containing both `.jsonl` and `.idx` files. Then run:

```bash
export MUMO_TRAIN_JSONL=/path/to/mumo_train.jsonl
export MUMO_BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
bash baselines/mumo_fresh/submit.sh
```

After training validates, set the frozen gate, original test JSON, evaluator
root, oracle cache, Python environments, base model, and final adapter, then
run:

```bash
bash baselines/mumo_fresh/submit_raw1.sh
```

Outputs default to `outputs/baselines/mumo_fresh/stable_v2_seed_32002`. The full job is
submitted with an `afterok` dependency on the smoke test, and a final CPU job
independently verifies that every saved adapter tensor is finite.

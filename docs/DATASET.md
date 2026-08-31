# MolProgramInstruct dataset

MolProgramInstruct is a unified instruction dataset for molecular construction
and source-conditioned molecular editing.

## Frozen release composition

| Mode | Records | Property-count buckets |
| --- | ---: | --- |
| De novo construction | 2,000,000 unique targets | 2p-7p |
| Source-conditioned editing | 569,919 unique source-target pairs | 1p-7p |

The construction pool is derived from PubChem bulk structures. The editing pool
is isolated from the MolEdit-Instruct training split. Each mode is sampled
uniformly across its property-count buckets. Editing pairs are never repeated to
fill a scarce bucket.

## Row contract

Each JSONL record contains:

```json
{
  "example_id": "stable identifier",
  "task_mode": "de_novo or edit",
  "source": "<EMPTY> or canonical source SMILES",
  "condition_program": [
    {"property": "MW", "goal": {"around": 320.0}}
  ],
  "target_smiles": "canonical target SMILES",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

The target is training supervision only. It is excluded from prompts,
generation-time inputs, candidate selection, and RL reward calculation.

## Leakage controls

- Canonical molecules overlapping frozen de novo evaluation targets are removed.
- Editing data are read only from the training split.
- Canonical source and target hashes are checked against frozen editing references.
- The manifest records SHA256 hashes, seeds, unique counts, duplicate counts,
  bucket distributions, and source-similarity distributions.

## Build

Construction and editing are built independently and then treated as one
release root:

```bash
python scripts/build_dataset.py de_novo \
  --input data/pubchem.jsonl \
  --output-dir data/molprogram/de_novo \
  --target-rows 2000000 \
  --heldout data/frozen_denovo.jsonl \
  --manifest data/molprogram/manifest.denovo.json

python scripts/build_dataset.py edit \
  --input-csv data/moledit_train.csv \
  --output-dir data/molprogram/edit \
  --target-rows 569919 \
  --heldout data/frozen_edit.jsonl \
  --manifest data/molprogram/manifest.edit.json
```

The indexed trainer additionally requires a `RELEASE_COMPLETE` marker after the
two manifests have been independently audited.

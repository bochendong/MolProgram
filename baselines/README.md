# Baselines

External-data and task-specific comparison systems live here. Baselines are
kept separate from MolProgram ablations because they may use a different prompt
contract, output serialization, or training corpus.

| Baseline | Purpose |
|---|---|
| [MuMO fresh LoRA](mumo_fresh/) | Train an editing-only Qwen LoRA from the unadapted backbone on MuMOInstruct |

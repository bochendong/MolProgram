# Shared-property transfer

This ablation tests whether the construction half of joint training can be made
more helpful to editing by removing unrelated construction programs. It reuses
the exact editing subset and evaluation gates from the matched 10,000-example
joint-versus-specialist baseline.

The candidate adapter receives:

- the baseline's exact 10,000 editing examples;
- 10,000 de-novo examples whose programs contain only `MW`, `LogP`, `QED`,
  `HBA`, and `RB`;
- equal replay quotas for two- and three-property programs, the compositional
  arities shared by the editing benchmark and the construction corpus.

The primary endpoint is strict success on the five editing tasks whose
properties all occur in construction. Positive transfer requires at least a
two-point strict gain over the editing specialist while retaining validity
within two points. The full ten-task editing aggregate and the complete de-novo
gate remain guardrails and are always reported.

This first-stage experiment changes data routing only. It does not yet use
gradient surgery or reinforcement learning, which keeps the causal comparison
easy to interpret.

## Submit

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export RELEASE_ROOT=/path/to/molprogram-release
export BASELINE_ROOT=/path/to/matched-10k-baseline
bash ablations/shared_property_transfer/submit_slurm.sh
```

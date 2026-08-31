# Negative-refinement ablation

This ablation isolates the effect of train-only negative completions from the
effect of additional optimization.

Both arms start from the same frozen positive-SFT adapter and use the same
training rows, row order, optimizer, learning rate, number of updates, and seed:

- `positive_only`: the matched continuation uses chosen-completion loss only;
- `multi_negative`: the same continuation adds the registered margin losses for
  invalid, condition-mismatch, opposite-program, partial-program, and edit
  source-copy completions.

The unmatched-parenthesis corruption is a syntax control, not a general
chemical hard negative. It differs from the chosen output only at the end and
primarily teaches the policy to terminate a valid serialization instead of
continuing an open branch. It does not represent failures such as incorrect
valence, broken ring closure, or a valid molecule that violates the requested
properties.

The positive-completion contribution is normalized to total weight one per
unique request in both arms. This prevents requests with more available
negative types from receiving extra supervised weight. Missing donor-dependent
negative types are recorded rather than silently replaced.

## Evaluation contract

Evaluate both arms on the same frozen target-blind Raw@1 gates:

- de novo: strict success and validity for every 2p through 7p bucket;
- editing: strict success, validity, copy rate, source similarity, and
  property-only success for every registered editing task.

The primary result is the per-mode delta between `multi_negative` and
`positive_only`. A gain on one task accompanied by a loss on the other is
reported as interference, not as an overall improvement.

If the first comparison supports negative refinement, use the following
three-arm follow-up to isolate syntax from semantic supervision:

1. `positive_only`;
2. `semantic_negatives`, containing source-copy, mismatch, opposite, and
   partial-program completions;
3. `semantic_plus_syntax`, adding the invalid-corruption completion.

This follow-up is the only basis for claiming that invalid corruption itself is
useful. A mixed multi-negative improvement cannot establish that attribution.

The existing positive-SFT checkpoint may be used for a zero-cost diagnostic,
but it is not the final control because the multi-negative arm has received
additional updates. The paper-facing comparison must use the matched
positive-only continuation described above.

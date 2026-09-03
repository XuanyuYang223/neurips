# Integer and Permutation Study Synthesis

This document records the shared paper-level interpretation of the integer and
permutation experiments. It deliberately keeps their primary metrics separate:
the integer study measures extrapolation to longer token sequences, whereas the
permutation study measures transfer to held-out properties and the geometry of
task-free hidden representations.

## Common question

Both studies ask whether broader multitask pretraining produces more reusable
internal structure. They use the same causal Transformer and causal MLP model
families, base-100-style symbolic tokenization, answer-only language-model loss,
fixed nested task sets, and repeated model seeds. Their complementary outcomes
are:

1. hard zero-shot or length extrapolation;
2. low-shot adaptation to tasks excluded from base training;
3. linear decodability of unseen targets; and
4. representation similarity under controlled task relationships.

## Permutation evidence

The permutation repository now contains several increasingly controlled
experiments.

- The v3 study trained 48 Transformer/MLP models on a verified 10-million-record,
  20-task corpus. Exact execution of unseen operations remained near zero even
  when teacher-forced token accuracy increased.
- The Property32 study trained 30 Transformers across three joint
  task-split/model-seed replicates and disjoint 16-property pools. Opposite-pool
  exact accuracy rose from 12.25% at `k=1` to 16.72% at `k=16`, but remained
  below a task-specific majority baseline.
- Linear probing found that final-layer opposite-pool information became more
  decodable through `k=8` and then declined slightly at `k=16`. This is evidence
  for a progressive internal signal, not a monotonic scaling law.
- Twenty-shot adaptation improved progressively with `k`, but the primary
  pretrained models used a lower learning rate than the random controls. The
  completed validation-only sensitivity shows that the matched `1e-5`
  pretrained-minus-random contrast changes from -5.30 points at `k=1` to
  +11.71 points at `k=16`; at matched `3e-4`, it changes from -0.36 to only
  +2.96 points and is non-monotonic between endpoints. The adaptation trend is
  therefore real under one optimization regime but not optimization-invariant.
- Controlled CKA shows that known combinatorial transformations produce
  stronger alignment than identity or incorrect-transform controls. Merely
  adding related tasks at a fixed task count did not yield a monotonic CKA dose
  response.
- A three-seed `k=16` 2x2 data-by-model-depth factorial is in progress to test
  whether the weak structured-holdout result is limited by data, depth, or their
  interaction.

## Integer evidence

The public `integer-multitask` branch reports a four-million-record corpus over
eight training tasks and nested `T1/T2/T4/T8` experiments. Its strict OOD target
is seven-to-ten-digit input, which requires more base-100 tokens than the
training distribution.

- Both architectures can learn several IID tasks well, but neither demonstrates
  reliable seven-to-ten-digit mathematical generalization. Some nonzero
  generated accuracy remains below task-specific constant-answer baselines.
- Six-digit accuracy should not be treated as token-length generalization:
  six-digit numbers can retain a familiar three-token base-100 width.
- The current twenty-shot transfer report covers seed 17. Its strongest result
  is transfer from the T8 `successor` task to held-out `predecessor`; this is
  evidence for direct task relatedness, not for task count alone.
- LCM, sorting, and modular-addition transfer is weak or absent, and the current
  report correctly identifies the need for the remaining seeds and explicit
  random controls.

The public source snapshot used here is commit
[`366f14f`](https://github.com/EmilyMeng05/Integer_task/tree/366f14f1c5ca65e718b35caf959cc0a35a5957ed).
Its stage-specific [T8 report](https://github.com/EmilyMeng05/Integer_task/blob/integer-multitask/docs/integer/README_T8_RESULTS.md)
and [few-shot report](https://github.com/EmilyMeng05/Integer_task/blob/integer-multitask/docs/integer/README_FEW_SHOT_TRANSFER.md)
are newer than its top-level progress summary and are therefore the sources of
record for this synthesis.

## Combined result

The current evidence does not support the claim that more training tasks alone
produce reliable hard generalization. The more defensible result is that
transfer is selective:

- direct mathematical relationships can support strong transfer or alignment;
- weaker internal signals may be visible to probes even when exact generation
  fails;
- apparent task-count trends are often non-monotonic and can be confounded by
  task identity, per-task exposure, optimization, and output difficulty; and
- multiple task-split/model-seed replicates and matched controls are necessary
  before interpreting a trend as a diversity effect.

This leads to the paper-level question:

> When hard systematic generalization fails, which forms of mathematical
> structure remain recoverable through low-shot adaptation, linear probes, and
> representation geometry?

## Reporting rules

The manuscript should follow these rules.

1. Report complete-answer exact accuracy as the primary behavioral metric.
2. Keep teacher-forced token accuracy, formatting validity, and loss as
   secondary diagnostics.
3. Compare every accuracy with an appropriate constant or random-init baseline.
4. Average within each model over a preregistered task set, then report the mean
   and sample standard deviation over independent joint replicates.
5. Do not pool model seeds, task subsets, and evaluation examples as if they
   were independent replicates.
6. Keep validation-only sensitivity analyses separate from frozen test results.
7. Describe CKA as representation alignment, not automatically as better
   representations or better generalization.

## Deadline-scoped remaining work

1. Finish, audit, and test the 12 missing permutation scaling models.
2. Export the scaling factorial effects with all raw model endpoints.
3. Add final figures and paper-ready Methods, Results, and Limitations text.
4. On the integer side, repeat the most informative twenty-shot comparisons for
   seeds 42 and 314159 before reporting error bars.

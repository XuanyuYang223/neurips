# Known-Relation Versus Unrelated-Pair CKA

## Research question

High CKA between two task specialists does not automatically imply that
multitask learning has discovered a general mathematical representation. For
example, descents and recoils are related by permutation inversion: recoils of
`pi` are descents of `pi^-1`. A high descent-versus-recoil CKA value may
therefore recover a real combinatorial relationship, but it may also reflect
that the two tasks are nearly the same computation under a simple input
transformation.

The more informative question is:

> Does representation similarity distinguish task pairs with a known
> combinatorial relationship from task pairs with no obvious duality?

This makes CKA a diagnostic of mathematical structure rather than treating
larger CKA as an unconditional success criterion.

## Frozen comparison groups

The primary known-related pairs are:

- `descents` versus `recoils`, related by permutation inversion;
- `peaks` versus `valleys`, related by value complementation.

The primary control pairs use the same four tasks but cross the two dual
families:

- `descents` versus `peaks`;
- `descents` versus `valleys`;
- `recoils` versus `peaks`;
- `recoils` versus `valleys`.

The pair list is fixed before viewing these CKA results. Control pairs are
called **no-obvious-duality** pairs, not independent pairs: absence of a known
simple duality is not proof of mathematical independence.

## Matched design

Each comparison uses independently trained, single-task, four-layer causal
Transformers with the same architecture, optimization schedule, data source,
permutation encoding, model seed, and number of updates. Exact duplicate
task/seed training designs share one canonical checkpoint; they are not
retrained or counted as independent replicates.

For every model seed, all specialists are evaluated on the same 4,096
task-free validation prefixes. Activations are extracted at `<ONE_END>`, before
the task token is supplied. Linear CKA is computed at every corresponding
layer, with final-layer CKA as the primary endpoint. The test split is not
used.

The unit of replication is the matched model seed. The primary analysis first
averages the four control-pair CKAs within each seed, then compares that value
with the mean of the two known-related-pair CKAs for the same seed. It reports:

- every pair-by-seed CKA value, without averaging it away;
- mean and sample standard deviation across the three seeds;
- the paired difference `CKA_known_related - CKA_no_obvious_duality`;
- layerwise curves and a paired permutation or sign test, labeled exploratory
  because three seeds provide low statistical power.

## Interpretation

If known-related pairs consistently have higher CKA than the matched controls,
the result supports the claim that representation geometry can recover known
combinatorial relationships. It does not by itself establish a universal
permutation representation, causal algorithm sharing, or behavioral transfer.

If both groups have similar CKA, the earlier increasing-`k` trend may reflect
shared inputs, architecture, optimization, or generic formatting rather than
specific mathematical relations. If control pairs are higher, the proposed
duality explanation is not supported and the individual pair results must be
examined before drawing a broader conclusion.

This pair analysis complements the relation-controlled multitask experiment:
the pair study asks whether known relationships are visible in specialist
geometry, while the multitask study asks whether increasing task diversity
raises similarity after obvious duals and high label correlations are
controlled.

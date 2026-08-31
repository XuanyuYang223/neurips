# Henry permutation experiments

There are now two deliberately separate experiment generations.  The original
v2 matrix is a completed baseline; the revised v3 matrix incorporates Henry
Kvinge's task-selection and category-comparison feedback.  Results from the two
must not be mixed under one protocol label.

## Completed v2 baseline

The frozen baseline config is
[`configs/henry_permutation.toml`](configs/henry_permutation.toml).  Its 20
tasks include `power`, `conjugate`, and `commutator`.

Twenty tasks were deterministically shuffled once.  The first 16 formed the
training pool, and the final four were held out from training.  Nested
training subsets contained 1, 2, 4, 8, and 16 tasks.  Both architectures were
trained with three independent parameter/data-order seeds:

```text
5 task subsets x 2 architectures x 3 seeds = 30 completed v2 models
```

The unaveraged validation results in the README and
`MODEL_TASK_ACCURACIES.csv` belong only to this v2 baseline.

## Henry's revision

Henry's feedback changes the main study in three ways:

1. Keep the architecture standard.  Reproducing the original PermuFormer
   architecture is not required; the existing small pre-LN decoder-only
   Transformer remains the Transformer condition.
2. Remove `power`, `conjugate`, and `commutator` from the main suite because
   their higher learning cost conflicts with the goal of training several
   quick, small models.
3. Compare learned representations after training only on
   encoding/translation, only on statistics/properties, or only on algebraic
   operations.

Henry proposed the three removals and the category comparison; he did not
select replacement tasks.  To retain exactly 20 balanced tasks, this project
chose the following replacements:

| Removed task | Added task | Definition | Answer type |
|---|---|---|---|
| `power` | `peaks` | Number of internal positions `i` with `pi[i-1] < pi[i] > pi[i+1]` | scalar |
| `conjugate` | `exceedances` | Number of positions `i` with `pi(i) > i` | scalar |
| `commutator` | `recoils` | Number of descents of `pi^-1` | scalar |

Each added property has exactly 500,000 verified records.  With every retained
task also at 500,000 records, v3 is a balanced 10,000,000-record corpus:

```text
20 tasks x 500,000 records/task = 10,000,000 records
```

These replacements are linear-time scalar targets with values below 30 for
the supported `n <= 30`, so they fit the existing atomic `00`--`99` number
encoding without `<NUM_START>` or additional operands.

### Completed v3 dataset

The formal v3 corpus was generated and fully verified before model training.
All revised models have since completed; data and model completion remain
separately authenticated artifacts.

| Artifact fact | Value |
|---|---:|
| Records | 10,000,000 |
| Shards | 100 |
| Compressed bytes | 1,139,175,228 |
| Generation time | 43.76 seconds |
| Full verification time | 34.59 seconds |

Parent manifest SHA-256:

```text
b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f
```

| Split | Records | Manifest SHA-256 |
|---|---:|---|
| Train (`000-097`) | 9,800,000 | `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa` |
| Validation (`098`) | 100,000 | `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068` |
| Test (`099`) | 100,000 | `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b` |

## Revised nested matrix (v3)

The design is frozen in
[`configs/henry_permutation_revised.toml`](configs/henry_permutation_revised.toml).
It points to protocol `permutation-20/v3` and
`data/permutation-10m-v3/manifest.json`.

The canonical v3 task list is shuffled with seed `20260830`, after which the
same four preregistered v2 holdout identities are moved to the end while all
other relative order is preserved.  This permits direct v2/v3 holdout
comparisons and keeps an algebra task in the holdout set.  The resulting order
is:

```text
 1  recoils
 2  lis_length
 3  fixed_points
 4  to_inversion_vector
 5  pattern_avoidance
 6  lds_length
 7  right_multiply_simple
 8  inverse
 9  to_cycle
10  descents
11  length
12  rsk_shape
13  peaks
14  cycle_type
15  bruhat_leq
16  exceedances
17  to_reduced_word       holdout
18  compose               holdout
19  parity                 holdout
20  to_lehmer              holdout
```

The revised nested matrix keeps the old controlled shape:

```text
5 task subsets x 2 architectures x 3 seeds = 30 completed v3 models
```

All runs used the same optimizer-update budget and frozen training
hyperparameters.
The output directory is separate (`runs/henry-permutation-v3`) so no v2
checkpoint can be overwritten or mistaken for a revised result.

Formal model status: **30/30 completed and strictly audited**. The final
independent test results are reported in [V3_RESULTS.md](V3_RESULTS.md), with
all 600 nested model-task test cells included in
[`V3_TEST_MODEL_TASK_ACCURACIES.csv`](V3_TEST_MODEL_TASK_ACCURACIES.csv).

## Matched category comparison

The full categories contain 4 encoding, 12 statistics, and 4 algebraic tasks.
Comparing all of them directly would confound task family with task count.
This project operationalizes Henry's representation question as three
task-count-matched four-task conditions:

| Condition | Frozen tasks |
|---|---|
| Encoding E4 | `to_cycle`, `to_lehmer`, `to_inversion_vector`, `to_reduced_word` |
| Statistics S4 | `length`, `cycle_type`, `rsk_shape`, `pattern_avoidance` |
| Algebra A4 | `inverse`, `compose`, `right_multiply_simple`, `bruhat_leq` |

Each condition uses 4 tasks, 500,000 records per task, the same model sizes,
the same optimizer-update budget, and seeds `17`, `42`, and `314159`:

```text
3 categories x 2 architectures x 3 seeds = 18 completed category models
```

The completed accuracy matrix evaluates behavioral cross-category transfer on
identical one-line input distributions. The separate representation analyses
remain planned: their primary landmark will be the hidden state at `<ONE_END>`,
before the task token, followed by layerwise CKA, frozen linear probes, and
few-shot cross-category transfer.

The `category_comparison` table in the revised TOML freezes this design, but it
is now executable through `permutation-experiments --matrix category`.  Its 18
run IDs and output directories are disjoint from the 30 nested runs, and the
same completion-marker and strict-audit rules apply.

The Encoding condition contains long reduced-word targets.  To hold the
effective batch at 64 examples/update in every category, E4 uses a micro-batch
of 4 with 16-way gradient accumulation; S4 and A4 use 16 with 4-way
accumulation.  The selected S4 tasks span scalar, structured, and Boolean
statistics, but they are a frozen representative subset rather than a claim
about all 12 statistical tasks.

## Completion record

Every v3 run used resumable model, optimizer, scheduler, and RNG state. Each of
the 48 atomic `completed.json` files records the final step, config and manifest
hashes, validation metrics, and final-checkpoint checksum. Strict post-training
audits report 30/30 nested and 18/18 category runs passed, with zero incomplete
or failed runs. The frozen one-time shard099 evaluation then processed 100,000
examples per model, or 4.8 million model-examples total.

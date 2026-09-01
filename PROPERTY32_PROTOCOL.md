# Zero-Overlap 32-Property Pilot

## Research question

The existing nested-task CKA result is confounded by task overlap: a larger
model contains all tasks learned by the smaller model. This pilot asks a more
specific question:

> As the number of learned permutation properties increases, do two models
> trained on completely disjoint task sets learn more similar task-free
> representations of the same permutation?

Pool A and Pool B are disjoint at every task count `k = 1, 2, 4, 8, 16`.
Every target is an integer in `0..n`, encoded as exactly one base-100 token for
`n <= 30`. This controls answer type and answer length across all 32 tasks.

## Properties

All positions are one-based. Cycles include fixed points unless a definition
explicitly excludes them.

| Family | Task | Definition |
|---|---|---|
| Local | `descents` | Number of adjacent pairs with `p(i) > p(i+1)` |
| Local | `recoils` | Number of descents of the inverse permutation |
| Local | `peaks` | Interior positions with `p(i-1) < p(i) > p(i+1)` |
| Local | `valleys` | Interior positions with `p(i-1) > p(i) < p(i+1)` |
| Local | `double_ascents` | Strictly increasing contiguous triples |
| Local | `double_descents` | Strictly decreasing contiguous triples |
| Local | `successions` | Adjacent pairs with `p(i+1) = p(i) + 1` |
| Local | `adjacencies` | Adjacent pairs whose values differ by one |
| Positional | `fixed_points` | Positions with `p(i) = i` |
| Positional | `anti_fixed_points` | Positions with `p(i) = n + 1 - i` |
| Positional | `exceedances` | Positions with `p(i) > i` |
| Positional | `deficiencies` | Positions with `p(i) < i` |
| Positional | `left_to_right_maxima` | Upper records scanned from the left |
| Positional | `left_to_right_minima` | Lower records scanned from the left |
| Positional | `right_to_left_maxima` | Upper records scanned from the right |
| Positional | `right_to_left_minima` | Lower records scanned from the right |
| Cycle | `cycle_count` | Number of disjoint cycles |
| Cycle | `two_cycle_count` | Number of cycles of length two |
| Cycle | `three_cycle_count` | Number of cycles of length three |
| Cycle | `even_cycle_count` | Number of even-length cycles |
| Cycle | `odd_cycle_count` | Number of odd-length cycles |
| Cycle | `longest_cycle` | Maximum cycle length |
| Cycle | `shortest_cycle` | Minimum cycle length |
| Cycle | `nontrivial_cycle_count` | Number of cycles of length at least two |
| Global/run | `lis_length` | Longest increasing subsequence length |
| Global/run | `lds_length` | Longest decreasing subsequence length |
| Global/run | `longest_increasing_run` | Longest contiguous increasing run |
| Global/run | `longest_decreasing_run` | Longest contiguous decreasing run |
| Global/run | `global_descents` | Splits where every prefix value exceeds every suffix value |
| Global/run | `components` | Number of direct-sum indecomposable components |
| Global/run | `max_displacement` | Maximum of `abs(p(i) - i)` |
| Global/run | `displacement_one_count` | Positions with `abs(p(i) - i) = 1` |

The definitions have fixture tests, boundary tests, and exhaustive checks over
all permutations through `S_6`. Cross-property identities are also tested,
including fixed points + exceedances + deficiencies = `n`, and even cycles +
odd cycles = all cycles.

## Frozen task pools

The order is chosen so `k = 4, 8, 16` contains equal numbers of local,
positional, cycle, and global/run properties.

| k position | Pool A | Pool B |
|---:|---|---|
| 1 | `descents` | `recoils` |
| 2 | `fixed_points` | `anti_fixed_points` |
| 3 | `cycle_count` | `two_cycle_count` |
| 4 | `lis_length` | `lds_length` |
| 5 | `peaks` | `valleys` |
| 6 | `exceedances` | `deficiencies` |
| 7 | `three_cycle_count` | `even_cycle_count` |
| 8 | `longest_increasing_run` | `longest_decreasing_run` |
| 9 | `double_ascents` | `double_descents` |
| 10 | `left_to_right_maxima` | `left_to_right_minima` |
| 11 | `odd_cycle_count` | `shortest_cycle` |
| 12 | `global_descents` | `components` |
| 13 | `successions` | `adjacencies` |
| 14 | `right_to_left_maxima` | `right_to_left_minima` |
| 15 | `longest_cycle` | `nontrivial_cycle_count` |
| 16 | `max_displacement` | `displacement_one_count` |

## Data

- Schema: `permutation-properties-32/v1`
- Permutation length: 2 through 30
- Seed: 20260901
- Total: 16,000,000 records
- Per property: 500,000 records
- Physical layout: 200 deterministic gzip shards of 80,000 records
- Train: 15,680,000 records (490,000 per property)
- Validation: 160,000 records (5,000 per property)
- Test: 160,000 records (5,000 per property; not read for pilot selection)
- Compressed size: 1,306,942,428 bytes
- Parent manifest SHA-256:
  `2b6e92965f2070684dbc660087ad5aaa078c91f0c2aecac72a9c05b1999ca31d`

The production verifier recomputed all 16,000,000 mathematical answers from
their input permutations, reconstructed every Passage token sequence, checked
every gzip hash, and confirmed exactly 500,000 records for each property.

## Pilot models

The pilot trains ten independent Transformer checkpoints: Pool A and Pool B at
each `k = 1, 2, 4, 8, 16`, using seed 17. Each run uses a standard four-layer,
pre-LN causal Transformer with `d_model=256`, eight heads, FFN multiplier four,
dropout 0.1, tied embeddings, 128-token context, and 3,240,448 trainable
parameters. Training uses 20,000 AdamW
updates, effective batch size 64, peak learning rate 3e-4, 1,000 warmup updates,
cosine decay, bf16, and identical optimizer/update budgets.

This one-seed matrix is a trend pilot. A confirmatory study needs at least two
additional seeds per cell for error bars.

The fixed update budget gives every run about 1.28 million training examples,
so exposure per trained property falls from about 1.28 million at `k=1` to
about 80,000 at `k=16`. The pilot therefore varies property diversity and
per-property exposure together; a confirmatory study should include a
matched-per-property-exposure control.

## Behavioral controls

Every target is one scalar answer token followed by EOS, but the scalar answer
distributions are not uniform. On the exact 160 examples used for each final
validation metric, a constant predictor that emits the most common answer
ranges from 7.5% to 83.75% exact accuracy across properties. Behavioral results
therefore report both raw exact accuracy and exact accuracy minus this
task-specific majority baseline. Raw unseen-task accuracy alone must not be
interpreted as mathematical generalization.

## CKA analysis

The primary metric is biased linear CKA between Pool A and Pool B at equal `k`
using 4,096 deterministic validation prefixes. Hidden states are extracted at
`<ONE_END>`, before the task token appears, so no requested operation is in the
probe input. The preregistered trend is final-layer CKA versus `log2(k)`.

Within-pool alignment to the corresponding `k=16` model is retained only as an
explicitly labeled overlapping-task control. It is not primary evidence.

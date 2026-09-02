# Relation-Controlled Property CKA Study

## Question

The first zero-overlap study prevented identical task names from appearing in
Pool A and Pool B, but several cross-pool tasks were known mathematical duals,
including descents/recoils and LIS/LDS. This experiment asks a stricter
question:

> When independently trained Transformers learn task sets with no shared task
> names, no co-selected predefined duals, and low empirical cross-pool label
> correlation, does final-layer representation similarity still increase with
> the number of learned properties?

## Frozen selection procedure

The selection procedure was completed before training and uses no validation
or test labels.

1. Read the first 50,000 permutations from training shard 000.
2. Compute all 32 scalar properties for every permutation.
3. Standardize each property separately within each permutation length
   `n = 2, ..., 30`, removing the common length trend.
4. Require exactly one selected task from each of the 16 predefined natural
   dual pairs. A dual pair can therefore never be co-selected or split across
   the two pools.
5. For each candidate, assign eight selected tasks to Pool A and eight to Pool
   B, with two tasks from every property family in each pool.
6. Score 600,000 deterministic candidates using
   `max_abs + 0.35 * q95_abs + 0.15 * mean_abs` over the 64 cross-pool label
   correlations.
7. Retain three low-scoring candidates whose 16 pair states differ in at least
   ten positions.

The frozen selection code seed is `20260903`. Exact values and task orders are
in `configs/property32_relation_controlled.toml`.

## Selected splits

Each four-task prefix contains one local, positional, cycle, and global/run
property. The eight-task prefix contains two from every family.

| Split | Pool A | Pool B | Mean abs. cross correlation | 95th percentile | Maximum |
|---|---|---|---:|---:|---:|
| S0 | recoils, left-to-right minima, three-cycle count, LDS, adjacencies, right-to-left minima, even-cycle count, components | valleys, anti-fixed points, cycle count, longest increasing run, double ascents, deficiencies, shortest cycle, displacement-one count | 0.1279 | 0.3359 | 0.3662 |
| S1 | recoils, left-to-right maxima, nontrivial-cycle count, LIS, successions, right-to-left minima, two-cycle count, global descents | valleys, fixed points, odd-cycle count, longest decreasing run, double descents, deficiencies, shortest cycle, displacement-one count | 0.1296 | 0.3213 | 0.3747 |
| S2 | peaks, anti-fixed points, cycle count, longest decreasing run, double descents, deficiencies, shortest cycle, global descents | recoils, left-to-right minima, three-cycle count, LIS, adjacencies, right-to-left maxima, even-cycle count, displacement-one count | 0.1356 | 0.3401 | 0.3703 |

"Low correlation" is a quantitative control, not a claim of mathematical
independence. Residual relationships remain, but the largest measured
conditional cross-pool correlation is below 0.375, and none of the 16
predefined dual pairs is co-selected.

## Training matrix

The matrix crosses all three splits with all three model seeds `17`, `42`, and
`101`. Each cell compares Pool A and Pool B at
`k = 1, 2, 4, 8`:

- 3 task selections x 3 seeds x 2 pools x 4 task counts = 72 logical cell
  positions;
- 60 unique Transformer checkpoints, because 12 positions repeat an exact
  ordered task prefix and seed from an earlier cell;
- repeated positions reference the same canonical checkpoint and are never
  treated as independent evidence;
- no checkpoint is initialized from another value of `k`;
- every equal-`k` A-vs-B comparison has zero task-name overlap.

All runs use the previously fully verified 16M scalar-property corpus, but
filter it to the tasks assigned to that checkpoint. The untouched test split
is not read.

## Model and optimization

- standard four-layer pre-LN causal Transformer;
- `d_model=256`, eight attention heads, FFN width 1,024;
- 3,240,448 trainable parameters, tied embeddings, dropout 0.1;
- 128-token context and bf16;
- 20,000 AdamW updates, effective batch size 64;
- peak learning rate 3e-4, 1,000 warmup updates, cosine decay.

The data use the same Passage Math one-line encoding as the preceding property
study. Every target is one scalar base-100 token followed by EOS.

## Frozen CKA analysis

All models are evaluated on the same 4,096 task-free validation prefixes at
`<ONE_END>`, before any task token is supplied. The primary metric is
final-layer linear CKA between Pool A and Pool B at equal `k`.

Analysis proceeds in two stages:

1. Construct one four-point CKA curve for each split×seed cell.
2. Aggregate the nine cell-level values at each `k` as mean plus/minus sample
   standard deviation.

The primary endpoint is the paired `CKA(k=8) - CKA(k=1)` delta across nine
cells. The report also includes an exact two-sided sign test, per-cell Spearman
correlations, strict-monotonicity counts, and a two-way split/seed sum-of-
squares decomposition. Strict monotonicity is diagnostic rather than a
success criterion.

## Interpretation boundary

This design tests whether the earlier positive CKA association survives after
removing obvious dual-task and high-label-correlation explanations. It does
not prove the selected properties are mathematically independent, and CKA
alone does not demonstrate behavioral generalization or shared algorithms.

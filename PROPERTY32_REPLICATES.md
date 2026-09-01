# Three-replicate zero-overlap property study

## Purpose

The original 32-property pilot used one model seed and one Pool A/B split. Its
final-layer CKA peaked sharply at `k=8`, so that result could reflect either
optimization noise or an idiosyncratic task partition. This confirmatory
extension repeats the full A/B x `k = 1, 2, 4, 8, 16` matrix three times.

## Frozen replicates

| Replicate | Model seed | Pool split seed | Natural dual pairs across pools | Models |
|---|---:|---:|---:|---:|
| R0 | 17 | manually frozen original | 16/16 | 10 completed pilot models |
| R1 | 42 | 20260941 | 6/16 | 10 new models |
| R2 | 101 | 20261075 | 8/16 | 10 new models |

Each replicate partitions all 32 properties into disjoint 16-task pools.
Every four-task block in both pools contains exactly one local, positional,
cycle, and global/run property, so prefixes at `k=4,8,16` are family-balanced.
Across the three replicates, every property occurs in Pool A at least once and
in Pool B at least once. No property is permanently assigned to one side.

The exact pool orders and immutable data/config hashes are stored in:

- `configs/property32_zero_overlap_pilot.toml`
- `configs/property32_zero_overlap_r1.toml`
- `configs/property32_zero_overlap_r2.toml`

Their SHA-256 values at launch are, respectively:

- `5cd3fc692880d83773e7fc80c006d16367b7c3aeebcdfa678659bf7fa189b2b3`
- `5d7453fa2ffde847cc74813f83da8dabddccbc893132002084eeff3ac4524693`
- `1a1c1247555fbbb5e6b3df0df3216580a7acffb914639ad47b55a18079bdac18`

## Controlled training

All 30 models use the same 16M verified corpus, model architecture, tokenizer,
optimizer, schedule, and 20,000-update budget described in
`PROPERTY32_PROTOCOL.md`. Only the task partition/order and model seed vary
between replicates. Every model is initialized independently; no checkpoint is
continued from another value of `k`.

## Frozen aggregation

For behavior, each replicate first averages the 16 opposite-pool tasks within
Pool A and Pool B separately, then averages the two directions. The three
replicate-level values are summarized as mean plus/minus sample standard
deviation. Primary behavioral metrics are exact-sequence accuracy and exact
accuracy minus the task-specific majority baseline.

For representations, each replicate compares Pool A with Pool B at equal `k`
on the same 4,096 task-free validation prefixes at `<ONE_END>`. Final-layer
linear CKA is summarized as mean plus/minus sample standard deviation over the
three replicate pairs at each `k`. Layerwise values, within-pool overlap
controls, and random-initialization controls remain available as diagnostics.

The main questions are:

1. Does mean A-vs-B CKA increase consistently with `k`?
2. Does the original `k=8` peak reproduce under less dual-aligned splits?
3. Does majority-adjusted opposite-pool exact accuracy improve with `k`?

The validation split is used for this analysis. The 160,000-example test split
remains untouched until all aggregation code and interpretation rules are
frozen.

## Interpretation boundary

This efficient design gives three joint replicates, not a full factorial
decomposition of randomness. Model seed and task split change together, so the
sample standard deviation captures their combined variability. A full 3-seed
x 3-split experiment would require 90 models and would be needed to estimate
the two variance sources separately.


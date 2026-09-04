# Property32 fixed-seed task-subset extension

## Question

The original three-replicate zero-overlap result jointly changed the model seed
and the task partition. Its error bars therefore combine optimization variance
with task-subset variance. This post-hoc extension asks a narrower question:

> At a fixed initialization/data-order seed, how much does the CKA trend change
> when the two disjoint task pools are independently reselected?

## Frozen design

The extension adds two task partitions, R3 and R4. Together with the original
R0 partition, they form three task-subset replicates that all use model seed
`17`. Each replicate trains two Transformers, Pool A and Pool B, at
`k = 1, 2, 4, 8, 16`, for 10 models per partition and 30 models in the
fixed-seed comparison. R0 already exists; R3 and R4 therefore require 20 new
models.

Every pool contains 16 unique Property32 tasks. Each consecutive four-task
prefix block contains one local, one positional, one cycle, and one global/run
property. Pool A and Pool B are disjoint at every matched `k`. R3 and R4 each
place exactly 8 of the 16 preregistered natural-duality pairs across pools and
8 within pools. The task orders and split seeds are frozen in:

- [`configs/property32_zero_overlap_r3.toml`](configs/property32_zero_overlap_r3.toml)
- [`configs/property32_zero_overlap_r4.toml`](configs/property32_zero_overlap_r4.toml)

All architecture, data, optimization, and activation-extraction settings equal
the original R0 protocol: a four-layer, eight-head Transformer with
`d_model=256`, 20,000 updates, and linear CKA on 4,096 deterministic validation
prefixes at `<ONE_END>`. The Property32 test split is not used.

## Analysis

For each partition and each `k`, first average the two directions A-vs-B and
B-vs-A for behavioral transfer, while CKA is symmetric and is recorded once.
Then report the mean, sample standard deviation, minimum, and maximum across
R0/R3/R4. These fixed-seed error bars estimate sensitivity to the chosen task
partition; they must not be described as optimization-seed error bars.

The original R0/R1/R2 result remains separately reported because those three
replicates jointly vary seed and partition. Comparing the two summaries helps
diagnose whether the previously high variance is primarily associated with
task selection, although three observations per summary cannot cleanly
identify a variance component.

## Limitations

- This extension was designed after viewing the original validation CKA trend.
- The prefixes are nested, so increasing `k` also changes the identities and
  exposure of the trained tasks.
- All models receive 20,000 updates, so per-task exposure falls as `k` grows.
- A common seed reduces one source of variation but induces dependence across
  the three task-partition replicates.

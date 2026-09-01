# V3 representation similarity with linear CKA

This analysis tests whether increasing the number of nested training tasks
makes the learned permutation representation more reproducible across random
seeds. It does not compare raw parameters. Every model is frozen, receives
the same task-free one-line prefix, and is measured at `<ONE_END>` before a
task token appears.

## Conclusion

The final-layer results do **not** show a consistent monotonic increase in
representation similarity as k grows.

- Transformer CKA changes from 0.5376 at k=1 to
  0.6323 at k=16, but peaks at
  0.7740 for
  k=4 and rises in only 2/4 adjacent steps.
- MLP CKA changes only from 0.8255 to
  0.8334, with a pronounced minimum of
  0.4924 at k=4.
- Random-init final-layer CKA is already 0.8833
  for the Transformer and 0.7588 for the MLP. Shared
  token identities, sequence positions, and architecture therefore create a
  substantial similarity baseline even before training.
- The same-seed Transformer becomes progressively closer to its k=16 model
  as k increases, but this secondary comparison is confounded by increasing
  overlap with the nested k=16 task set. The MLP does not show that pattern.

The defensible observation is that task count changes representation geometry,
sometimes substantially, but the current single task order does not support
the claim that adding tasks generally makes independently trained models
converge to one common representation.

## Primary result: cross-seed stability

Values are mean +/- sample SD over the three seed pairs (17-42, 17-314159,
42-314159). Higher linear CKA means more similar representation geometry.

| Architecture | Trained tasks (k) | Final-layer CKA | Delta from random init |
|---|---:|---:|---:|
| Transformer | 1 | 0.5376 +/- 0.1938 | -0.3457 |
| Transformer | 2 | 0.5092 +/- 0.2364 | -0.3741 |
| Transformer | 4 | 0.7740 +/- 0.0708 | -0.1093 |
| Transformer | 8 | 0.6301 +/- 0.0355 | -0.2532 |
| Transformer | 16 | 0.6323 +/- 0.0358 | -0.2510 |
| MLP | 1 | 0.8255 +/- 0.0154 | +0.0667 |
| MLP | 2 | 0.8563 +/- 0.0324 | +0.0975 |
| MLP | 4 | 0.4924 +/- 0.1734 | -0.2664 |
| MLP | 8 | 0.5649 +/- 0.2153 | -0.1939 |
| MLP | 16 | 0.8334 +/- 0.0381 | +0.0746 |

Trend diagnostics across the five k values:

- Transformer: k=1 to k=16 change +0.0947; Spearman rho +0.600; 2/4 adjacent steps increased.
- MLP: k=1 to k=16 change +0.0079; Spearman rho -0.100; 3/4 adjacent steps increased.

A positive endpoint change alone is not evidence of a monotonic trend; the
Spearman and adjacent-step diagnostics show whether the intermediate k
values support the same direction.

## Secondary result: alignment with k=16

This compares each smaller model with the same-seed k=16 model. Because
the task sets are nested, increasing k also increases task overlap with the
reference. These values are descriptive and are not an isolated causal
effect of task count.

| Architecture | Smaller k | Final-layer CKA to k=16 |
|---|---:|---:|
| Transformer | 1 | 0.4266 +/- 0.0636 |
| Transformer | 2 | 0.5194 +/- 0.1236 |
| Transformer | 4 | 0.5953 +/- 0.1077 |
| Transformer | 8 | 0.6448 +/- 0.0689 |
| MLP | 1 | 0.6287 +/- 0.0349 |
| MLP | 2 | 0.6336 +/- 0.0600 |
| MLP | 4 | 0.5480 +/- 0.1627 |
| MLP | 8 | 0.6091 +/- 0.1746 |

## Random-initialization controls

| Architecture | Final-layer CKA |
|---|---:|
| Transformer | 0.8833 +/- 0.0038 |
| MLP | 0.7588 +/- 0.0163 |

## Exploratory cross-architecture comparison

These final-layer values compare the Transformer and MLP trained with the
same k and seed. They are exploratory because CKA is easiest to interpret
within a shared architecture.

| Trained tasks (k) | Transformer-MLP final-layer CKA |
|---:|---:|
| 1 | 0.4322 +/- 0.1799 |
| 2 | 0.5470 +/- 0.1056 |
| 4 | 0.5130 +/- 0.2244 |
| 8 | 0.4511 +/- 0.1660 |
| 16 | 0.5327 +/- 0.1443 |

## Protocol

- Probe split: validation shard 098; test shard 099 was not read.
- Probe examples: 4,096, selected deterministically
  with seed 20260831 and probe SHA-256 `41aa38be2f75c7d903467d90fa503547d92cea13a3b34385f217e3c210a09183`.
- Landmark: the hidden vector at `<ONE_END>` from the embedding output,
  every model block, and final layer normalization.
- Metric: biased linear CKA over examples, accumulated in float64.
- Reference implementation: [Ristori's `ckatorch`](https://github.com/RistoAle97/centered-kernel-alignment)
  at commit
  `f7e2aefee17b6440088d62830881ba30b797fe92`; the local Gram-free formula is
  regression-tested against direct centered-Gram CKA.
- Primary units: three pairwise comparisons among three independently
  trained seeds for each architecture and k.
- Random-init controls use the same architectures, tokenizer, positions,
  inputs, and seeds without training.
- Analysis implementation commit: `c598ce311f3f7e25dfe160e25fc86c78166b7a45`.

## Interpretation limits

Only one frozen nested task order was trained. Thus k is confounded with
which tasks were added and with reduced per-task exposure at larger k. The
three seed pairs are also dependent because each of three models appears in
two pairs. The results can establish an observed representation-stability
trend, but not a general causal law that task diversity creates a universal
permutation representation. Cross-architecture CKA is exploratory because
the Transformer and causal MLP have different computational structures.

## Machine-readable files

- `pairwise_layer_cka.csv`: every preregistered model/layer comparison.
- `summary.csv`: pair means, sample SDs, minima, and maxima.
- `probe_manifest.json`: exact probe IDs, length histogram, and data hash.
- `manifest.json`: model checkpoint hashes and output checksums.

# V3 disjoint-category representation similarity

This controlled CKA comparison uses the 18 completed category models. Every
model was trained on exactly four tasks. Encoding E4, Statistics S4, and
Algebra A4 are pairwise disjoint, so cross-condition task overlap is zero.
All models receive the same task-free validation prefixes and are measured at
`<ONE_END>` before any task token appears.

## Conclusion

The existing category models show a clear family-dependent representation
signal. Same-family models are substantially more similar across seeds than
models trained on disjoint families. Matching the initialization seed does
not recover the lost similarity between disjoint families:

- Transformer: same-family 0.5556; disjoint/different-seed 0.3221; disjoint/same-seed 0.3243.
- MLP: same-family 0.6419; disjoint/different-seed 0.2885; disjoint/same-seed 0.2818.

This supports proceeding to a larger controlled-overlap study. It does not
yet establish a causal task-family effect because family is still tied to
specific operations, output formats, and difficulty.

## Main comparison

The table aggregates final-layer comparisons across the three task families.
Within-condition pairs have 100% task overlap and different seeds. Disjoint
pairs have 0% task overlap and are reported separately for matched and
different seeds.

| Architecture | Comparison | Pairs | Final-layer CKA |
|---|---|---:|---:|
| Transformer | Same family, different seed | 9 | 0.5556 +/- 0.1246 |
| Transformer | Disjoint families, same seed | 9 | 0.3243 +/- 0.1017 |
| Transformer | Disjoint families, different seed | 18 | 0.3221 +/- 0.0998 |
| Transformer | Random init, different seed | 3 | 0.8833 +/- 0.0038 |
| MLP | Same family, different seed | 9 | 0.6419 +/- 0.1598 |
| MLP | Disjoint families, same seed | 9 | 0.2818 +/- 0.1657 |
| MLP | Disjoint families, different seed | 18 | 0.2885 +/- 0.1830 |
| MLP | Random init, different seed | 3 | 0.7588 +/- 0.0163 |

## Same-family cross-seed detail

| Architecture | Training family | Final-layer CKA |
|---|---|---:|
| Transformer | Encoding E4 | 0.6053 +/- 0.0684 |
| Transformer | Statistics S4 | 0.4159 +/- 0.0974 |
| Transformer | Algebra A4 | 0.6457 +/- 0.0527 |
| MLP | Encoding E4 | 0.5471 +/- 0.0923 |
| MLP | Statistics S4 | 0.8449 +/- 0.0080 |
| MLP | Algebra A4 | 0.5337 +/- 0.0280 |

## Disjoint-family same-seed detail

The two models in each row started from identical seed-specific weights;
their only formal experimental difference is the disjoint training family.

| Architecture | Family pair | Final-layer CKA |
|---|---|---:|
| Transformer | Encoding E4 vs Statistics S4 | 0.2275 +/- 0.0305 |
| Transformer | Encoding E4 vs Algebra A4 | 0.4209 +/- 0.0959 |
| Transformer | Statistics S4 vs Algebra A4 | 0.3246 +/- 0.0568 |
| MLP | Encoding E4 vs Statistics S4 | 0.4955 +/- 0.0596 |
| MLP | Encoding E4 vs Algebra A4 | 0.1683 +/- 0.0442 |
| MLP | Statistics S4 vs Algebra A4 | 0.1815 +/- 0.0368 |

## Interpretation

- Transformer: same-family cross-seed CKA is 0.5556; disjoint-family cross-seed CKA is 0.3221 (difference +0.2336); disjoint-family same-seed CKA is 0.3243.
- MLP: same-family cross-seed CKA is 0.6419; disjoint-family cross-seed CKA is 0.2885 (difference +0.3534); disjoint-family same-seed CKA is 0.2818.

This fixes k at four and removes task overlap, but task family remains
confounded with task identities, output types, and difficulty. It therefore
tests whether these three existing families produce different geometries;
it does not yet estimate a general causal effect of task count. A larger
controlled-overlap study would need multiple balanced task subsets at each k.
The reported SDs summarize correlated model pairs and are descriptive; they
are not independent-replicate confidence intervals.

## Protocol and files

- Probe: 4,096 validation-shard-098 prefixes; SHA-256
  `41aa38be2f75c7d903467d90fa503547d92cea13a3b34385f217e3c210a09183`. Test shard 099 was not read.
- Layers: embedding output, every model block, and final normalization.
- Metric: biased linear CKA over examples with float64 accumulation.
- Analysis implementation commit: `18290a397b1caafeb7c11348212f1381b7dc06df`.
- `category_pairwise_layer_cka.csv`: all individual comparisons.
- `category_summary.csv`: family-pair summaries.
- `category_overall_summary.csv`: primary aggregate summaries.
- `manifest.json`: exact checkpoints, probe identity, and artifact hashes.

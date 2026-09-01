# Three-replicate zero-overlap property results

The study contains 30 independently trained Transformers: three joint
task-split/model-seed replicates, two disjoint pools, and five values of
`k`. Values are mean plus/minus sample standard deviation across the three
replicate-level measurements. Test data were not used.

## Opposite-pool behavior

| k | Loss | Token accuracy | Exact accuracy | Exact minus majority |
|---:|---:|---:|---:|---:|
| 1 | 5.4587 +/- 1.5490 | 54.20% +/- 0.67% | 12.25% +/- 2.66% | -20.59 +/- 2.66 pp |
| 2 | 4.6739 +/- 0.7637 | 55.65% +/- 1.05% | 11.49% +/- 2.22% | -21.34 +/- 2.22 pp |
| 4 | 3.7018 +/- 0.3602 | 55.73% +/- 0.81% | 11.96% +/- 1.90% | -20.87 +/- 1.90 pp |
| 8 | 2.8060 +/- 0.4315 | 56.56% +/- 1.85% | 13.64% +/- 3.73% | -19.19 +/- 3.73 pp |
| 16 | 2.2308 +/- 0.0758 | 58.34% +/- 1.24% | 16.72% +/- 2.50% | -16.11 +/- 2.50 pp |

Exact unseen-property accuracy remains below the task-specific
majority baseline at every k. The behavioral result therefore does
not demonstrate reliable hard zero-shot execution.

## Final-layer A-vs-B linear CKA

| k | Mean +/- sample SD | Min | Max |
|---:|---:|---:|---:|
| 1 | 0.172313 +/- 0.073809 | 0.087759 | 0.223846 |
| 2 | 0.191957 +/- 0.119854 | 0.122691 | 0.330352 |
| 4 | 0.212761 +/- 0.092239 | 0.156898 | 0.319226 |
| 8 | 0.619041 +/- 0.216940 | 0.379959 | 0.803336 |
| 16 | 0.496115 +/- 0.150254 | 0.387884 | 0.667665 |

Mean-trend Spearman rho: 0.900000.
Mean-trend Pearson r against log2(k): 0.828033.
Replicates peaking at k=8: 2/3.
Mean CKA is monotonic non-decreasing: false.

The three replicates jointly vary model seed and task split. Their
sample SD therefore captures combined variability and does not separate
the two variance sources. The fixed-update per-task exposure confound
also remains. See the
[frozen protocol](../../../PROPERTY32_REPLICATES.md) for details.

## Artifacts

- [Replicate-level behavioral values](behavior_replicates.csv)
- [Behavioral mean and sample SD](behavior_summary.csv)
- [Replicate-level CKA values](cka_replicates.csv)
- [CKA mean, sample SD, minimum, and maximum](cka_summary.csv)
- Child reports: [R0](r0/behavior/README.md), [R1](r1/behavior/README.md), and [R2](r2/behavior/README.md)

# Fixed-seed task-subset replicate results

R0, R3, and R4 all use Transformer seed 17 while independently changing the family-balanced A/B task partition. The error bars therefore estimate task-subset sensitivity without mixing it with initialization-seed variability.

## Opposite-pool behavioral generalization

| k | Loss | Token accuracy | Exact accuracy | Exact minus majority |
|---:|---:|---:|---:|---:|
| 1 | 6.3285 ± 0.8062 | 54.38% ± 0.91% | 11.17% ± 2.71% | -21.66 ± 2.71 pp |
| 2 | 4.5692 ± 0.4587 | 53.82% ± 1.44% | 8.95% ± 1.85% | -23.89 ± 1.85 pp |
| 4 | 3.4346 ± 0.1706 | 54.96% ± 0.94% | 10.61% ± 1.48% | -22.23 ± 1.48 pp |
| 8 | 3.0015 ± 0.5089 | 54.89% ± 1.07% | 10.57% ± 2.23% | -22.27 ± 2.23 pp |
| 16 | 2.3192 ± 0.0785 | 58.15% ± 0.65% | 16.35% ± 1.37% | -16.48 ± 1.37 pp |

## Final-layer linear CKA between disjoint pools

| k | Mean ± sample SD | Min | Max |
|---:|---:|---:|---:|
| 1 | 0.100583 ± 0.112056 | 0.004875 | 0.223846 |
| 2 | 0.175244 ± 0.135487 | 0.080007 | 0.330352 |
| 4 | 0.302321 ± 0.159643 | 0.134899 | 0.452839 |
| 8 | 0.643951 ± 0.154096 | 0.495752 | 0.803336 |
| 16 | 0.600428 ± 0.180319 | 0.432796 | 0.791200 |

Mean-trend Spearman rho is 0.900; the k=16 minus k=1 CKA change is +0.500.

These three measurements are task-split replicates, not three independent random initializations. They should be reported separately from R0/R1/R2, which jointly varied both split and model seed.
Validation data were used; the test split was not read.

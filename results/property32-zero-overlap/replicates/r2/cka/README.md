# Zero-overlap 32-property CKA pilot

The primary comparison uses independently trained Pool A and Pool B
Transformers at equal k. The pools have zero task overlap at every k,
all targets are one scalar token, and activations are extracted at
`<ONE_END>` before any task token is supplied.

Probe examples: 4,096 deterministic validation prefixes.

| k | Final-layer linear CKA (A vs B) |
|---:|---:|
| 1 | 0.087759 |
| 2 | 0.122691 |
| 4 | 0.156898 |
| 8 | 0.379959 |
| 16 | 0.667665 |

Spearman rho across k: 1.000000.
Pearson r against log2(k): 0.920322.
k=16 minus k=1: +0.579906.

The sequence is not monotonic: the largest value occurs at `k=8`,
followed by a substantial decline at `k=16`. Thus the pilot shows a
positive descriptive association, not stable convergence as tasks grow.

## Controls

Random-initialization cross-seed final-layer CKA: 0.885069.

| Pool | k vs 16 final-layer CKA |
|---|---:|
| A 1 vs A 16 | 0.382970 |
| A 2 vs A 16 | 0.211608 |
| A 4 vs A 16 | 0.210151 |
| A 8 vs A 16 | 0.436060 |
| B 1 vs B 16 | 0.253907 |
| B 2 vs B 16 | 0.426560 |
| B 4 vs B 16 | 0.572356 |
| B 8 vs B 16 | 0.646441 |

The within-pool rows are overlapping-task controls and are not primary
zero-overlap evidence. The random baseline is also not a performance
baseline: high CKA can arise from common architecture and input geometry.

Pool A and Pool B share no task names, but several properties are natural
duals (for example descents/recoils and LIS/LDS). The `k=8` spike may
therefore reflect conceptual symmetry rather than generic task diversity.

This is a one-seed pilot. It establishes a descriptive trend, not
an error-bar-supported population claim. Test data were not used.

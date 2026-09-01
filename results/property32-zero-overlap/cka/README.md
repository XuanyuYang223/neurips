# Zero-overlap 32-property CKA pilot

The primary comparison uses independently trained Pool A and Pool B
Transformers at equal k. The pools have zero task overlap at every k,
all targets are one scalar token, and activations are extracted at
`<ONE_END>` before any task token is supplied.

Probe examples: 4,096 deterministic validation prefixes.

| k | Final-layer linear CKA (A vs B) |
|---:|---:|
| 1 | 0.223846 |
| 2 | 0.330352 |
| 4 | 0.319226 |
| 8 | 0.803336 |
| 16 | 0.432796 |

Spearman rho across k: 0.800000.
Pearson r against log2(k): 0.624110.
k=16 minus k=1: +0.208950.

The sequence is not monotonic: the largest value occurs at `k=8`,
followed by a substantial decline at `k=16`. Thus the pilot shows a
positive descriptive association, not stable convergence as tasks grow.

## Controls

Random-initialization cross-seed final-layer CKA: 0.885069.

| Pool | k vs 16 final-layer CKA |
|---|---:|
| A 1 vs A 16 | 0.126486 |
| A 2 vs A 16 | 0.230086 |
| A 4 vs A 16 | 0.400720 |
| A 8 vs A 16 | 0.442274 |
| B 1 vs B 16 | 0.286516 |
| B 2 vs B 16 | 0.552547 |
| B 4 vs B 16 | 0.427748 |
| B 8 vs B 16 | 0.787674 |

The within-pool rows are overlapping-task controls and are not primary
zero-overlap evidence. The random baseline is also not a performance
baseline: high CKA can arise from common architecture and input geometry.

Pool A and Pool B share no task names, but several properties are natural
duals (for example descents/recoils and LIS/LDS). The `k=8` spike may
therefore reflect conceptual symmetry rather than generic task diversity.

This is a one-seed pilot. It establishes a descriptive trend, not
an error-bar-supported population claim. Test data were not used.

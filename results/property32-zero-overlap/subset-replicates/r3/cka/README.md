# Zero-overlap 32-property CKA pilot

The primary comparison uses independently trained Pool A and Pool B
Transformers at equal k. The pools have zero task overlap at every k,
all targets are one scalar token, and activations are extracted at
`<ONE_END>` before any task token is supplied.

Probe examples: 4,096 deterministic validation prefixes.

| k | Final-layer linear CKA (A vs B) |
|---:|---:|
| 1 | 0.073029 |
| 2 | 0.115373 |
| 4 | 0.134899 |
| 8 | 0.632766 |
| 16 | 0.791200 |

Spearman rho across k: 1.000000.
Pearson r against log2(k): 0.918306.
k=16 minus k=1: +0.718170.

The sequence is not monotonic: the largest value occurs at `k=8`,
followed by a substantial decline at `k=16`. Thus the pilot shows a
positive descriptive association, not stable convergence as tasks grow.

## Controls

Random-initialization cross-seed final-layer CKA: 0.885069.

| Pool | k vs 16 final-layer CKA |
|---|---:|
| A 1 vs A 16 | 0.250637 |
| A 2 vs A 16 | 0.575875 |
| A 4 vs A 16 | 0.842727 |
| A 8 vs A 16 | 0.712599 |
| B 1 vs B 16 | 0.256018 |
| B 2 vs B 16 | 0.232606 |
| B 4 vs B 16 | 0.153059 |
| B 8 vs B 16 | 0.805573 |

The within-pool rows are overlapping-task controls and are not primary
zero-overlap evidence. The random baseline is also not a performance
baseline: high CKA can arise from common architecture and input geometry.

Pool A and Pool B share no task names, but several properties are natural
duals (for example descents/recoils and LIS/LDS). The `k=8` spike may
therefore reflect conceptual symmetry rather than generic task diversity.

This is a one-seed pilot. It establishes a descriptive trend, not
an error-bar-supported population claim. Test data were not used.

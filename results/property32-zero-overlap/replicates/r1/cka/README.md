# Zero-overlap 32-property CKA pilot

The primary comparison uses independently trained Pool A and Pool B
Transformers at equal k. The pools have zero task overlap at every k,
all targets are one scalar token, and activations are extracted at
`<ONE_END>` before any task token is supplied.

Probe examples: 4,096 deterministic validation prefixes.

| k | Final-layer linear CKA (A vs B) |
|---:|---:|
| 1 | 0.205335 |
| 2 | 0.122829 |
| 4 | 0.162158 |
| 8 | 0.673828 |
| 16 | 0.387884 |

Spearman rho across k: 0.600000.
Pearson r against log2(k): 0.637840.
k=16 minus k=1: +0.182548.

The sequence is not monotonic: the largest value occurs at `k=8`,
followed by a substantial decline at `k=16`. Thus the pilot shows a
positive descriptive association, not stable convergence as tasks grow.

## Controls

Random-initialization cross-seed final-layer CKA: 0.885069.

| Pool | k vs 16 final-layer CKA |
|---|---:|
| A 1 vs A 16 | 0.546493 |
| A 2 vs A 16 | 0.484229 |
| A 4 vs A 16 | 0.491845 |
| A 8 vs A 16 | 0.792964 |
| B 1 vs B 16 | 0.128396 |
| B 2 vs B 16 | 0.043514 |
| B 4 vs B 16 | 0.063874 |
| B 8 vs B 16 | 0.367730 |

The within-pool rows are overlapping-task controls and are not primary
zero-overlap evidence. The random baseline is also not a performance
baseline: high CKA can arise from common architecture and input geometry.

Pool A and Pool B share no task names, but several properties are natural
duals (for example descents/recoils and LIS/LDS). The `k=8` spike may
therefore reflect conceptual symmetry rather than generic task diversity.

This is a one-seed pilot. It establishes a descriptive trend, not
an error-bar-supported population claim. Test data were not used.

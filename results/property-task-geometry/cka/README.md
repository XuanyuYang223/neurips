# Combinatorial task-geometry CKA results

All values below use the same 4,096 task-free validation prefixes.
The test split was not read.

## Single-task geometry

| Comparison | Task-pair units | Final-layer CKA, mean +/- sample SD |
|---|---:|---:|
| same_task | 16 | 0.709551 +/- 0.213933 |
| direct_relation | 8 | 0.137482 +/- 0.145448 |
| no_direct_relation | 112 | 0.090325 +/- 0.057124 |

Task-label permutation contrast (direct minus other): +0.047157; one-sided p=0.015050.
Random-initialization final-layer CKA across seeds: 0.879503 +/- 0.010841.
Directly related tasks are more similar than the other cross-task pairs, but the absolute direct-relation CKA is far below the same-task value. The high random-initialization baseline reflects shared architecture and input geometry, not learned task structure.

## Fixed-four-task composition

| Direct correspondences r | Cells | Final-layer CKA, mean +/- sample SD |
|---:|---:|---:|
| 0 | 12 | 0.293518 +/- 0.198795 |
| 1 | 12 | 0.275232 +/- 0.171298 |
| 2 | 12 | 0.266206 +/- 0.171843 |
| 4 | 12 | 0.364759 +/- 0.258605 |

Mean paired r=4 minus r=0 delta: +0.071241 +/- 0.212117.
Positive cells: 7/12; monotonic cells: 1/12.
Two-sided exact sign-test p for r=4 minus r=0: 0.774414.
This controlled experiment does not support a monotonic CKA dose-response as direct correspondences increase: the effect is strongly heterogeneous across bundle layouts.

## Symmetry-aligned mechanism

Correct minus identity CKA: +0.383165 +/- 0.251034.
Correct minus wrong-transform CKA: +0.415232 +/- 0.232887.
Descriptively, both contrasts are positive in 24/24 and 24/24 pair-seed units. Treating those units as independent would give two-sided exact sign-test p-values of 1.192e-07 and 1.192e-07; these are descriptive because seeds are clustered within relations.
After averaging over seeds, both contrasts remain positive in 8/8 and 8/8 mathematical relations. The primary relation-level two-sided exact sign-test p-values are 0.007812 and 0.007812.
The strongest result is therefore transformation-specific: models trained on known dual properties align when their inputs are related by the corresponding combinatorial symmetry.

CKA is a representation diagnostic, not a behavioral accuracy metric or proof of a shared algorithm. See the raw CSV files for every model, task pair, seed, condition, and layer.

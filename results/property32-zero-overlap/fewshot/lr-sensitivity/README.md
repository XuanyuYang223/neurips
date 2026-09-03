# Property32 twenty-shot matched-learning-rate sensitivity

This post-hoc analysis uses validation only. Values are exact sequence accuracy mean +/- sample SD over three joint task-split/model-seed replicates.

| Learning rate | k | Pretrained | Random init | Pretrained minus random |
|---:|---:|---:|---:|---:|
| 1e-05 | 1 | 16.77% +/- 2.68% | 22.07% +/- 7.97% | -5.30 +/- 7.71 pp |
| 1e-05 | 2 | 20.70% +/- 2.14% | 22.07% +/- 7.97% | -1.37 +/- 9.66 pp |
| 1e-05 | 4 | 26.03% +/- 1.23% | 22.07% +/- 7.97% | +3.96 +/- 8.42 pp |
| 1e-05 | 8 | 31.90% +/- 4.34% | 22.07% +/- 7.97% | +9.82 +/- 4.50 pp |
| 1e-05 | 16 | 33.78% +/- 3.59% | 22.07% +/- 7.97% | +11.71 +/- 5.24 pp |
| 3e-04 | 1 | 34.14% +/- 2.77% | 34.51% +/- 4.00% | -0.36 +/- 3.55 pp |
| 3e-04 | 2 | 35.55% +/- 1.32% | 34.51% +/- 4.00% | +1.04 +/- 2.71 pp |
| 3e-04 | 4 | 36.48% +/- 2.57% | 34.51% +/- 4.00% | +1.97 +/- 1.80 pp |
| 3e-04 | 8 | 35.65% +/- 3.10% | 34.51% +/- 4.00% | +1.14 +/- 2.58 pp |
| 3e-04 | 16 | 37.46% +/- 2.45% | 34.51% +/- 4.00% | +2.96 +/- 1.67 pp |

## Paired learning-rate effects

Positive values mean that `3e-4` achieved higher exact accuracy than `1e-5`. The interaction is `(pretrained high - pretrained low) - (random high - random low)`.

| k | Pretrained high minus low | Random high minus low | Initialization x LR |
|---:|---:|---:|---:|
| 1 | +17.37 +/- 1.21 pp | +12.43 +/- 4.10 pp | +4.94 +/- 5.15 pp |
| 2 | +14.85 +/- 3.24 pp | +12.43 +/- 4.10 pp | +2.42 +/- 7.28 pp |
| 4 | +10.44 +/- 3.02 pp | +12.43 +/- 4.10 pp | -1.99 +/- 6.95 pp |
| 8 | +3.75 +/- 1.26 pp | +12.43 +/- 4.10 pp | -8.68 +/- 2.86 pp |
| 16 | +3.68 +/- 1.97 pp | +12.43 +/- 4.10 pp | -8.75 +/- 4.30 pp |

## Interpretation

At the matched `1e-5` learning rate, the pretrained-minus-random contrast changes from -5.30 percentage points at `k=1` to +11.71 points at `k=16`. At the matched `3e-4` learning rate, it changes from -0.36 to +2.96 points and is not monotonic across the intermediate task counts.

The progressive low-learning-rate curve therefore survives a matched-initialization comparison, but its magnitude is optimization-dependent. The high learning rate substantially improves both random initialization and small-k warm starts, compressing the apparent k trend. With only three joint replicates, these validation-only contrasts should be reported as a sensitivity analysis rather than a new confirmatory test result.

The learning-rate sweep was specified after the primary test result and does not reuse the Property32 test split.

# Four-representation transfer results

Three jointly trained Transformers saw the one-line row and descents column (11 cells).
The other 21 representation-task combinations received no gradient updates.

| Status | Cells | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus majority |
|---|---:|---:|---:|---:|---:|---:|
| seen | 11 | 0.513 ± 0.010 | 78.20 ± 0.49% | 54.31 ± 0.96% | 14.97% | +39.34 ± 0.96 pp |
| held_out | 21 | 1.457 ± 0.183 | 67.11 ± 1.29% | 30.61 ± 2.64% | 19.15% | +11.46 ± 2.64 pp |

Means are cell-macro averages computed within each seed, followed by mean ± sample SD over three seeds.
Exact-sequence accuracy is the primary complete-answer metric; token accuracy is teacher-forced.

## Exact-sequence accuracy by cell

Each entry is mean ± sample SD over the three seeds. Cells marked `*` were used for training.

| Representation | length | parity | peaks | exceedances | fixed_points | descents | recoils | lis_length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one_line | 30.76 ± 1.29%* | 53.49 ± 0.11%* | 51.27 ± 0.58%* | 87.63 ± 4.13%* | 94.25 ± 1.59%* | 44.69 ± 2.05%* | 42.88 ± 1.04%* | 58.79 ± 1.62%* |
| cycle | 9.14 ± 0.62% | 51.84 ± 0.85% | 35.71 ± 3.52% | 32.13 ± 0.91% | 29.56 ± 2.78% | 40.37 ± 0.11%* | 38.87 ± 0.03% | 26.15 ± 4.99% |
| lehmer | 7.53 ± 0.82% | 48.48 ± 5.33% | 22.08 ± 16.88% | 30.85 ± 1.95% | 28.06 ± 6.56% | 49.28 ± 1.75%* | 40.80 ± 0.65% | 26.48 ± 6.10% |
| inversion_vector | 7.24 ± 1.03% | 52.29 ± 0.28% | 23.93 ± 13.08% | 29.97 ± 1.99% | 32.20 ± 1.23% | 43.96 ± 0.71%* | 43.25 ± 0.42% | 26.31 ± 4.14% |

## Exact accuracy minus the constant-answer majority baseline

| Representation | length | parity | peaks | exceedances | fixed_points | descents | recoils | lis_length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| one_line | +27.32 ± 1.29 pp* | +3.21 ± 0.11 pp* | +39.93 ± 0.58 pp* | +79.87 ± 4.13 pp* | +57.39 ± 1.59 pp* | +37.05 ± 2.05 pp* | +34.92 ± 1.04 pp* | +42.35 ± 1.62 pp* |
| cycle | +5.70 ± 0.62 pp | +1.56 ± 0.85 pp | +24.37 ± 3.52 pp | +24.37 ± 0.91 pp | -7.30 ± 2.78 pp | +32.73 ± 0.11 pp* | +30.91 ± 0.03 pp | +9.71 ± 4.99 pp |
| lehmer | +4.09 ± 0.82 pp | -1.80 ± 5.33 pp | +10.74 ± 16.88 pp | +23.09 ± 1.95 pp | -8.80 ± 6.56 pp | +41.64 ± 1.75 pp* | +32.84 ± 0.65 pp | +10.04 ± 6.10 pp |
| inversion_vector | +3.80 ± 1.03 pp | +2.01 ± 0.28 pp | +12.59 ± 13.08 pp | +22.21 ± 1.99 pp | -4.66 ± 1.23 pp | +36.32 ± 0.71 pp* | +35.29 ± 0.42 pp | +9.87 ± 4.14 pp |

[MODEL_CELL_ACCURACIES.csv](MODEL_CELL_ACCURACIES.csv) contains all 96 unaveraged model-cell rows; [CELL_SUMMARY.csv](CELL_SUMMARY.csv) contains loss, token accuracy, and exact accuracy for every cell.

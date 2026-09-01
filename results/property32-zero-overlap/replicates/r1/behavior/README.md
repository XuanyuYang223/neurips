# Behavioral results: zero-overlap 32-property pilot

Ten independently trained Transformers are compared: Pool A and Pool B
at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap
with its training set. All metrics below are task-macro averages on the
diagnostic validation split; the held-back test split was not used.

## Opposite-pool transfer

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 4.9255 | 53.11% | 11.21% | 34.80% | -23.59 pp |
| A | 2 | 4.2305 | 55.66% | 11.33% | 34.80% | -23.48 pp |
| A | 4 | 4.5378 | 55.80% | 12.03% | 34.80% | -22.77 pp |
| A | 8 | 2.4307 | 57.09% | 14.30% | 34.80% | -20.51 pp |
| A | 16 | 2.2531 | 57.58% | 15.16% | 34.80% | -19.65 pp |
| B | 1 | 2.5685 | 55.84% | 16.13% | 30.86% | -14.73 pp |
| B | 2 | 4.3204 | 57.95% | 16.52% | 30.86% | -14.34 pp |
| B | 4 | 2.8448 | 57.29% | 15.66% | 30.86% | -15.20 pp |
| B | 8 | 2.8484 | 58.24% | 16.48% | 30.86% | -14.37 pp |
| B | 16 | 2.0500 | 56.29% | 12.58% | 30.86% | -18.28 pp |

## Opposite-pool average across A and B

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.7470 | 54.47% | 13.67% | 32.83% | -19.16 pp |
| 2 | 4.2754 | 56.81% | 13.93% | 32.83% | -18.91 pp |
| 4 | 3.6913 | 56.54% | 13.85% | 32.83% | -18.98 pp |
| 8 | 2.6395 | 57.67% | 15.39% | 32.83% | -17.44 pp |
| 16 | 2.1516 | 56.93% | 13.87% | 32.83% | -18.96 pp |

## Seen-task performance

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.2791 | 87.81% | 75.62% | 11.25% | +64.38 pp |
| A | 2 | 0.2177 | 89.53% | 79.06% | 19.06% | +60.00 pp |
| A | 4 | 0.2744 | 87.97% | 75.94% | 30.00% | +45.94 pp |
| A | 8 | 0.4097 | 81.45% | 62.89% | 26.33% | +36.56 pp |
| A | 16 | 0.5139 | 78.01% | 56.02% | 30.86% | +25.16 pp |
| B | 1 | 0.5200 | 79.06% | 58.13% | 32.50% | +25.63 pp |
| B | 2 | 0.2404 | 88.91% | 77.81% | 34.06% | +43.75 pp |
| B | 4 | 0.2837 | 88.05% | 76.09% | 54.84% | +21.25 pp |
| B | 8 | 0.3614 | 85.20% | 70.39% | 43.12% | +27.27 pp |
| B | 16 | 0.4860 | 79.84% | 59.69% | 34.80% | +24.88 pp |

`token_accuracy` is teacher-forced accuracy over the scalar answer token
and EOS. `sequence_accuracy` requires both tokens to be correct and is the
primary complete-answer metric. `MODEL_TASK_ACCURACIES.csv` contains every
unaveraged model-task result; `SUMMARY.csv` contains the task-macro values.
The majority baseline always predicts each task's most common answer on the
same 160 examples, exposing gains that are only answer-frequency guessing.

This is a one-seed pilot with a fixed 20,000-update budget. Per-task
exposure therefore falls as k increases, so any trend mixes task diversity
with reduced examples per learned task. It is descriptive, not a
population-level claim with error bars.

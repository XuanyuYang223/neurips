# Behavioral results: zero-overlap 32-property pilot

Ten independently trained Transformers are compared: Pool A and Pool B
at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap
with its training set. All metrics below are task-macro averages on the
diagnostic validation split; the held-back test split was not used.

## Opposite-pool transfer

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 4.1142 | 53.32% | 15.20% | 33.79% | -18.59 pp |
| A | 2 | 5.1997 | 54.59% | 9.38% | 33.79% | -24.41 pp |
| A | 4 | 3.9784 | 53.89% | 7.77% | 33.79% | -26.02 pp |
| A | 8 | 2.4445 | 58.54% | 17.07% | 33.79% | -16.72 pp |
| A | 16 | 2.1710 | 58.89% | 17.77% | 33.79% | -16.02 pp |
| B | 1 | 7.6156 | 56.07% | 12.58% | 31.87% | -19.30 pp |
| B | 2 | 5.9092 | 56.19% | 12.54% | 31.87% | -19.34 pp |
| B | 4 | 4.1560 | 55.98% | 12.30% | 31.87% | -19.57 pp |
| B | 8 | 2.5205 | 56.66% | 15.27% | 31.87% | -16.60 pp |
| B | 16 | 2.4339 | 59.65% | 19.30% | 31.87% | -12.58 pp |

## Opposite-pool average across A and B

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.8649 | 54.70% | 13.89% | 32.83% | -18.95 pp |
| 2 | 5.5545 | 55.39% | 10.96% | 32.83% | -21.88 pp |
| 4 | 4.0672 | 54.93% | 10.04% | 32.83% | -22.79 pp |
| 8 | 2.4825 | 57.60% | 16.17% | 32.83% | -16.66 pp |
| 16 | 2.3025 | 59.27% | 18.54% | 32.83% | -14.30 pp |

## Seen-task performance

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.3560 | 85.00% | 70.00% | 40.00% | +30.00 pp |
| A | 2 | 0.0948 | 95.94% | 91.88% | 25.00% | +66.88 pp |
| A | 4 | 0.2346 | 90.47% | 80.94% | 29.69% | +51.25 pp |
| A | 8 | 0.3364 | 86.41% | 72.81% | 37.27% | +35.55 pp |
| A | 16 | 0.4515 | 81.04% | 62.07% | 31.87% | +30.20 pp |
| B | 1 | 0.0002 | 100.00% | 100.00% | 9.38% | +90.62 pp |
| B | 2 | 0.0030 | 100.00% | 100.00% | 18.75% | +81.25 pp |
| B | 4 | 0.2961 | 86.80% | 73.59% | 32.34% | +41.25 pp |
| B | 8 | 0.4162 | 82.23% | 64.45% | 38.05% | +26.41 pp |
| B | 16 | 0.5376 | 76.88% | 53.75% | 33.79% | +19.96 pp |

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

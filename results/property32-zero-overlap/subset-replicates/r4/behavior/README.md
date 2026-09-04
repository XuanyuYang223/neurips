# Behavioral results: zero-overlap 32-property pilot

Ten independently trained Transformers are compared: Pool A and Pool B
at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap
with its training set. All metrics below are task-macro averages on the
diagnostic validation split; the held-back test split was not used.

## Opposite-pool transfer

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 6.4063 | 56.78% | 13.55% | 37.19% | -23.63 pp |
| A | 2 | 4.5110 | 52.48% | 9.53% | 37.19% | -27.66 pp |
| A | 4 | 2.8630 | 57.66% | 15.70% | 37.19% | -21.48 pp |
| A | 8 | 2.0608 | 56.80% | 15.43% | 37.19% | -21.76 pp |
| A | 16 | 2.1856 | 57.30% | 14.61% | 37.19% | -22.58 pp |
| B | 1 | 7.2399 | 53.73% | 14.96% | 28.48% | -13.52 pp |
| B | 2 | 4.3607 | 51.84% | 4.18% | 28.48% | -24.30 pp |
| B | 4 | 3.7879 | 52.85% | 5.86% | 28.48% | -22.62 pp |
| B | 8 | 2.7669 | 55.43% | 10.86% | 28.48% | -17.62 pp |
| B | 16 | 2.4628 | 57.70% | 15.43% | 28.48% | -13.05 pp |

## Opposite-pool average across A and B

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.8231 | 55.25% | 14.26% | 32.83% | -18.57 pp |
| 2 | 4.4358 | 52.16% | 6.86% | 32.83% | -25.98 pp |
| 4 | 3.3255 | 55.25% | 10.78% | 32.83% | -22.05 pp |
| 8 | 2.4139 | 56.11% | 13.14% | 32.83% | -19.69 pp |
| 16 | 2.3242 | 57.50% | 15.02% | 32.83% | -17.81 pp |

## Seen-task performance

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.0025 | 100.00% | 100.00% | 20.00% | +80.00 pp |
| A | 2 | 0.0572 | 98.91% | 97.81% | 27.81% | +70.00 pp |
| A | 4 | 0.3485 | 84.69% | 69.38% | 27.66% | +41.72 pp |
| A | 8 | 0.5852 | 74.49% | 48.98% | 29.14% | +19.84 pp |
| A | 16 | 0.5309 | 77.34% | 54.69% | 28.48% | +26.21 pp |
| B | 1 | 0.0001 | 100.00% | 100.00% | 44.38% | +55.62 pp |
| B | 2 | 0.2294 | 90.47% | 80.94% | 27.50% | +53.44 pp |
| B | 4 | 0.2335 | 90.78% | 81.56% | 50.00% | +31.56 pp |
| B | 8 | 0.3949 | 85.08% | 70.16% | 38.36% | +31.80 pp |
| B | 16 | 0.4557 | 81.15% | 62.30% | 37.19% | +25.12 pp |

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

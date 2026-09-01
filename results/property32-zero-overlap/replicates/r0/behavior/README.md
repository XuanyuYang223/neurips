# Behavioral results: zero-overlap 32-property pilot

Ten independently trained Transformers are compared: Pool A and Pool B
at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap
with its training set. All metrics below are task-macro averages on the
diagnostic validation split; the held-back test split was not used.

## Opposite-pool transfer

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 8.3333 | 54.63% | 11.52% | 34.84% | -23.32 pp |
| A | 2 | 4.4065 | 55.76% | 11.52% | 34.84% | -23.32 pp |
| A | 4 | 3.2294 | 57.83% | 15.66% | 34.84% | -19.18 pp |
| A | 8 | 3.3403 | 54.20% | 8.44% | 34.84% | -26.41 pp |
| A | 16 | 2.5956 | 59.84% | 19.69% | 34.84% | -15.16 pp |
| B | 1 | 5.1949 | 52.25% | 6.84% | 30.82% | -23.98 pp |
| B | 2 | 3.9773 | 53.75% | 7.66% | 30.82% | -23.16 pp |
| B | 4 | 3.4647 | 53.59% | 8.32% | 30.82% | -22.50 pp |
| B | 8 | 3.2515 | 54.65% | 10.27% | 30.82% | -20.55 pp |
| B | 16 | 1.8811 | 57.77% | 15.82% | 30.82% | -15.00 pp |

## Opposite-pool average across A and B

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.7641 | 53.44% | 9.18% | 32.83% | -23.65 pp |
| 2 | 4.1919 | 54.76% | 9.59% | 32.83% | -23.24 pp |
| 4 | 3.3470 | 55.71% | 11.99% | 32.83% | -20.84 pp |
| 8 | 3.2959 | 54.42% | 9.36% | 32.83% | -23.48 pp |
| 16 | 2.2383 | 58.81% | 17.75% | 32.83% | -15.08 pp |

## Seen-task performance

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.0002 | 100.00% | 100.00% | 9.38% | +90.62 pp |
| A | 2 | 0.1244 | 95.00% | 90.00% | 26.88% | +63.13 pp |
| A | 4 | 0.4062 | 81.09% | 62.19% | 25.00% | +37.19 pp |
| A | 8 | 0.4212 | 81.84% | 63.67% | 29.61% | +34.06 pp |
| A | 16 | 0.4960 | 79.43% | 58.87% | 30.82% | +28.05 pp |
| B | 1 | 0.1867 | 91.88% | 83.75% | 10.62% | +73.12 pp |
| B | 2 | 0.4759 | 81.25% | 62.50% | 27.19% | +35.31 pp |
| B | 4 | 0.3634 | 84.92% | 69.84% | 34.84% | +35.00 pp |
| B | 8 | 0.4466 | 80.70% | 61.41% | 30.55% | +30.86 pp |
| B | 16 | 0.4796 | 79.94% | 59.88% | 34.84% | +25.04 pp |

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

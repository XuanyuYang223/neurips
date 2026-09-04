# Behavioral results: zero-overlap 32-property pilot

Ten independently trained Transformers are compared: Pool A and Pool B
at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap
with its training set. All metrics below are task-macro averages on the
diagnostic validation split; the held-back test split was not used.

## Opposite-pool transfer

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 4.0792 | 56.35% | 15.00% | 30.74% | -15.74 pp |
| A | 2 | 4.1841 | 56.48% | 15.62% | 30.74% | -15.12 pp |
| A | 4 | 3.8746 | 53.71% | 9.88% | 30.74% | -20.86 pp |
| A | 8 | 3.4743 | 52.99% | 7.85% | 30.74% | -22.89 pp |
| A | 16 | 2.7315 | 57.46% | 14.92% | 30.74% | -15.82 pp |
| B | 1 | 6.7173 | 52.54% | 5.16% | 34.92% | -29.77 pp |
| B | 2 | 5.9754 | 52.58% | 5.16% | 34.92% | -29.77 pp |
| B | 4 | 3.3878 | 54.10% | 8.20% | 34.92% | -26.72 pp |
| B | 8 | 3.1150 | 55.27% | 10.55% | 34.92% | -24.38 pp |
| B | 16 | 2.0589 | 58.81% | 17.62% | 34.92% | -17.30 pp |

## Opposite-pool average across A and B

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.3983 | 54.44% | 10.08% | 32.83% | -22.75 pp |
| 2 | 5.0798 | 54.53% | 10.39% | 32.83% | -22.44 pp |
| 4 | 3.6312 | 53.91% | 9.04% | 32.83% | -23.79 pp |
| 8 | 3.2947 | 54.13% | 9.20% | 32.83% | -23.63 pp |
| 16 | 2.3952 | 58.13% | 16.27% | 32.83% | -16.56 pp |

## Seen-task performance

| Pool | k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 0.0788 | 98.44% | 96.88% | 43.75% | +53.12 pp |
| A | 2 | 0.2008 | 92.81% | 85.62% | 43.75% | +41.88 pp |
| A | 4 | 0.4911 | 77.03% | 54.06% | 34.22% | +19.84 pp |
| A | 8 | 0.4526 | 80.59% | 61.17% | 34.14% | +27.03 pp |
| A | 16 | 0.5327 | 78.03% | 56.05% | 34.92% | +21.13 pp |
| B | 1 | 0.0010 | 100.00% | 100.00% | 7.50% | +92.50 pp |
| B | 2 | 0.0848 | 96.41% | 92.81% | 23.75% | +69.06 pp |
| B | 4 | 0.2875 | 87.73% | 75.47% | 25.16% | +50.31 pp |
| B | 8 | 0.3342 | 86.25% | 72.50% | 22.97% | +49.53 pp |
| B | 16 | 0.4416 | 80.88% | 61.76% | 30.74% | +31.02 pp |

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

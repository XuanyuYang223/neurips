# Deadline-Scoped k=16 Scaling Protocol

## Question

At the fully populated `k=16` endpoint, is weak unseen-task accuracy primarily
limited by the amount of training data, model depth, or their interaction?

The existing 1x-data/1x-model v3 runs form the baseline. Three interventions
produce a 2x2 factorial design:

| Condition | Training exposure | Transformer layers | MLP blocks |
|---|---:|---:|---:|
| Baseline | 1.28M examples | 4 | 1 |
| 10x data, 1x model | 12.8M examples | 4 | 1 |
| 1x data, 2x model | 1.28M examples | 8 | 2 |
| 10x data, 2x model | 12.8M examples | 8 | 2 |

Each cell contains Transformer and MLP models with seeds 17, 42, and 314159,
for 24 endpoints. Six baseline models already exist. The three intervention
cells had seed-17 pilots; this protocol completes seeds 42 and 314159, adding
12 models. It intentionally does not claim to complete the earlier 90-model
all-`k` scaling proposal.

## Frozen outcomes

The primary outcome is the task-macro complete-answer sequence accuracy over
the three structured training holdouts: `to_reduced_word`, `compose`, and
`to_lehmer`. Boolean `parity` is reported separately because its output length
and baseline difficulty are not comparable. Loss and teacher-forced token
accuracy are secondary diagnostics.

Metrics are first averaged over the three structured tasks within each model.
The report then gives mean and sample standard deviation over three seeds for
each architecture and factorial condition. Data, depth, and interaction
contrasts are paired by architecture and seed.

All cells use the same v3 task order, `k=16` training prefix, optimizer,
effective batch size, and frozen validation/test examples. The original
163-token v3 vocabulary is recovered from the dataset manifest so later
Property32 tokenizer extensions cannot change model size.

The machine-readable protocol is
[`configs/permutation_scaling_k16.toml`](configs/permutation_scaling_k16.toml).
Training, audit, one-time evaluation, and aggregation are controlled by
`permutation-scaling-k16`.

## Completed result

All 24 endpoints completed training and passed strict checkpoint audit. Each
model was evaluated once on the same frozen 100,000-record v3 test split, with
5,000 examples per task. Exact accuracy on each primary structured holdout
(`to_reduced_word`, `compose`, and `to_lehmer`) is 0% for every architecture,
condition, and seed. Consequently, all paired effects on the primary exact
macro are 0.000 +/- 0.000 percentage points. Loss and teacher-forced token
accuracy vary by architecture and condition but do not provide evidence of
successful operation-level transfer.

The complete results, error-bar figure, unaveraged endpoints, paired contrasts,
and paper-ready text are in
[`results/v3/scaling/k16/`](results/v3/scaling/k16/README.md).

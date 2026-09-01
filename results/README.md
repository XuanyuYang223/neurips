# Experiment results

Results are separated by dataset and experiment protocol so that the v2
baseline cannot be mistaken for the revised v3 study.

## v2 baseline

- [Report](v2/README.md)
- [All 600 model-task validation rows](v2/model_task_accuracies.csv)

V2 trained 30 models: 2 architectures x 5 nested task counts x 3 seeds. Each
model was evaluated on all 20 tasks, giving 30 x 20 = 600 rows. These are
validation results, not an untouched final test evaluation.

## v3 revised study

- [Main report](v3/README.md)
- [Launch and provenance record](v3/LAUNCH.md)
- [Henry-style 20-shot follow-up protocol](v3/FEW_SHOT_PROTOCOL.md)
- [Henry-style 20-shot results](v3/fewshot/README.md)
- [Layerwise CKA representation analysis](v3/cka/README.md)
- [Disjoint-category CKA follow-up](v3/cka/category/README.md)
- [Data and model scaling protocol](v3/SCALING_PROTOCOL.md)
- [All 960 independent-test model-task rows](v3/test_model_task_accuracies.csv)
- [Nested generalization only](v3/test_nested_generalization.csv)
- [Category generalization only](v3/test_category_generalization.csv)

V3 trained 48 models: 30 nested models and 18 category-comparison models.
Every model was evaluated on all 20 tasks, giving 48 x 20 = 960 rows on
validation and another 960 rows on the frozen test split.

The separate Henry-style follow-up contains 120 warm-start adaptations and 24
random-init controls. Each is evaluated only on its assigned holdout task, so
its raw table has 144 rows rather than another 20-task grid.

The CKA package uses 4,096 task-free prefixes from validation shard 098 to
compare hidden-state geometry across the 30 nested base models. It contains
285 pairwise layer comparisons, summary statistics, random-initialization
controls, exact probe IDs, and checkpoint/data provenance. It does not read or
reuse final test predictions.

The two `*_generalization.csv` files intentionally exclude tasks used to train
the evaluated model. The other summary files retain seen-task metrics for
diagnostics and direct confirmation that training succeeded.

## Metric conventions

- `loss` is task-macro mean answer-token negative log likelihood; lower is
  better.
- `token_accuracy` is teacher-forced and can be inflated by delimiters,
  copied tokens, and answer format.
- `sequence_accuracy` requires the complete canonical answer and EOS to be
  correct; it is the primary operation-level accuracy.
- Means are first computed across tasks within one run, then reported as mean
  and sample standard deviation across the three seeds.

Raw accuracy values in CSV files are fractions in `[0, 1]`; Markdown reports
display percentages.

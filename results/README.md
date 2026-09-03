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
- [Deadline-scoped k=16 scaling protocol](../SCALING_K16.md)
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

## 32-property zero-overlap study

- [Combined report](property32-zero-overlap/README.md)
- [Three-replicate aggregate](property32-zero-overlap/replicates/README.md)
- [Replicate-level behavioral values](property32-zero-overlap/replicates/behavior_replicates.csv)
- [Replicate-level CKA values](property32-zero-overlap/replicates/cka_replicates.csv)
- [Original R0 behavioral results](property32-zero-overlap/behavior/README.md)
- [Original R0 layerwise CKA results](property32-zero-overlap/cka/README.md)
- [Layerwise linear-probing results](property32-zero-overlap/linear-probing/README.md)
- [All 6,336 unaveraged model-property-layer probe rows](property32-zero-overlap/linear-probing/model_task_layer_probes.csv)
- [Twenty-shot fine-tuning results](property32-zero-overlap/fewshot/README.md)
- [All 144 unaveraged fine-tuning endpoints](property32-zero-overlap/fewshot/model_task_results.csv)

This exploratory extension uses 32 scalar permutation properties divided into
two disjoint 16-task pools. Three joint task-split/model-seed replicates train
Pool A and Pool B at `k = 1, 2, 4, 8, 16`, producing 30 Transformers and 960
validation model-task rows. Behavioral exact accuracy is compared with a
per-task majority-answer baseline because several property distributions are
highly imbalanced. CKA uses the same 4,096 task-free validation prefixes for
all models and does not read the test split.

The linear-probing follow-up reads task-free `<ONE_END>` activations from the
same 30 trained Transformers and three random-initialization controls. Ridge
probes are fitted and tuned on 4,096 validation permutations, then evaluated
once on 4,096 independently selected property-test permutations. Its primary
question is whether values of the 16 opposite-pool properties become more
linearly decodable as base-training task count increases.

The twenty-shot follow-up adapts each of the 30 base Transformers to four
balanced properties from the opposite pool, yielding 120 warm starts and 24
support-matched random-init controls. Exact accuracy and paired gains from
zero-shot increase with `k`, but the pretrained models do not exceed the
random-init control on average. Its final endpoint uses all 2,500 examples per
property from source shard 199, disjoint from the linear-probe sample.

## Combinatorial task-geometry study

- [Main CKA report](property-task-geometry/cka/README.md)
- [Every specialist-model comparison](property-task-geometry/cka/specialist_pairwise_cka.csv)
- [Every fixed-four-task comparison](property-task-geometry/cka/bundle_cell_cka.csv)
- [Every symmetry-control comparison](property-task-geometry/cka/symmetry_cka.csv)
- [Checkpoint and run provenance](property-task-geometry/cka/run_provenance.csv)

This confirmatory study contains 48 single-task specialists and 60
fixed-four-task Transformers, totaling 108 audited checkpoints. CKA uses the
same 4,096 task-free validation prefixes for every comparison and does not
read the test split. Directly related specialist pairs are modestly more
similar than other cross-task pairs, and the mathematically correct inverse
or complement produces a large, consistent alignment advantage. The
fixed-task-count bundle experiment does not establish a monotonic increase in
CKA as more direct mathematical correspondences are added. Symmetry inference
uses the eight mathematical relations as its primary units after aggregating
the three seeds; all eight relation-level contrasts are positive
(`p=0.0078125`).

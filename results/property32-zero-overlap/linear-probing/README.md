# Permutation linear-probing results

Linear ridge probes read task-free `<ONE_END>` activations from the
30 completed zero-overlap Transformers. Probes were trained and tuned
on 4,096 validation permutations and evaluated once
on 4,096 independent test permutations.

## Opposite-pool final-layer results

Each value first macro-averages the 16 unseen opposite-pool properties
within a model, then the two pool directions within a replicate, and
finally reports mean +/- sample SD across three joint task-split/model-seed
replicates.

| k | Length-conditioned R2 | Exact accuracy | Length-mode baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|
| 1 | 0.1978 +/- 0.0770 | 45.49% +/- 2.75% | 42.26% | +3.23 pp |
| 2 | 0.2447 +/- 0.0343 | 47.35% +/- 1.17% | 42.26% | +5.09 pp |
| 4 | 0.2706 +/- 0.0154 | 48.14% +/- 0.67% | 42.26% | +5.89 pp |
| 8 | 0.3069 +/- 0.0174 | 49.69% +/- 0.55% | 42.26% | +7.43 pp |
| 16 | 0.2968 +/- 0.0141 | 49.31% +/- 0.71% | 42.26% | +7.05 pp |

Final-layer R2 Spearman rho across k: 0.9000.
k=16 minus k=1 R2: +0.0990.
Best mean R2 occurs at k=8; monotonic non-decreasing: false.

The main result is a positive but non-monotonic trend. All three replicates
improve from `k=1` to `k=8`. From `k=8` to `k=16`, one replicate improves and
two regress, so the evidence does not support a monotonic dose-response claim.
At the property level, 24 of 32 target-property means are higher at `k=16`
than at `k=1`; properties are descriptive targets, not independent statistical
replicates.

## Random-initialization control

Final-layer length-conditioned R2: 0.2148 +/- 0.0024.
Final-layer exact accuracy: 46.26% +/- 0.10%.

Relative to the random control, mean final-layer R2 changes by -0.0170,
+0.0299, +0.0559, +0.0921, and +0.0820 for
`k = 1, 2, 4, 8, 16`. Exact accuracy changes by -0.77, +1.09, +1.88, +3.42,
and +3.05 percentage points. Every `k=8` and `k=16` replicate exceeds its
seed-matched random control in both metrics, whereas the mean `k=1` model does
not. Thus much of the absolute decodability is supplied by the architecture's
random feature map, but multitask training adds a smaller, reproducible signal.

The embedding-layer length-conditioned R2 is numerically zero for trained and
random models, as expected after removing the length-only conditional mean.
The decodable signal emerges inside the Transformer blocks, which is a useful
sanity check against direct target or length leakage at the input landmark.

R2 is measured after removing the training-set conditional mean and
scale within each permutation length. Exact accuracy rounds the linear
prediction back to an integer property value. The raw CSV retains every
model, target property, layer, selected ridge coefficient, and metric.

This is a linear decodability result, not evidence that the base model
can behaviorally execute an unseen operation or that the decoded feature
is causally used by the model.

## Files

- `model_task_layer_probes.csv`: all 6,336 unaveraged
  `(model, property, layer)` results for 30 trained and three random models;
- `model_macro_probes.csv`: per-model task-macro results separated into
  trained, same-pool-untrained, and opposite-pool targets;
- `opposite_pool_replicates.csv`: the two pool directions averaged within each
  of the three joint task-split/model-seed replicates;
- `opposite_pool_summary.csv`: mean and sample SD across the three replicates;
- `random_baseline_summary.csv`: mean and sample SD across three randomly
  initialized Transformers;
- `probe_manifests.json`, `run_provenance.json`, and `manifest.json`: exact
  sample identities, checkpoint hashes, implementation commit, and artifact
  hashes.

`length_conditioned_r2` is the primary metric. `pearson_r`, normalized RMSE,
rounded integer exact accuracy, the length-conditional modal baseline, the
selected ridge coefficient, and validation tuning score are retained in the
CSV files. The statistical sample size for the primary error bars is three,
and model seed changes jointly with the frozen task-pool split.

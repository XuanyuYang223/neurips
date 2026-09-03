# Permutation Linear-Probing Protocol

## Question

Does increasing the number of permutation properties used in base-model
training make information about unseen properties more linearly recoverable?

The analysis uses the completed 30-model, three-replicate zero-overlap
Transformer matrix. Each replicate partitions the same 32 scalar permutation
properties into disjoint 16-task pools A and B. Models were trained at
`k = 1, 2, 4, 8, 16` in each pool. For an A model, all 16 B properties are
unseen probe targets, and conversely for a B model.

The frozen machine-readable protocol is
[`configs/property32_linear_probe.toml`](configs/property32_linear_probe.toml).

## Representation and labels

Every base model receives the same task-free one-line permutation prefix.
Layerwise residual-stream vectors are extracted at `<ONE_END>`, before a task
token or answer is present. This prevents task-token and answer leakage.

All 32 integer labels are recomputed from the permutation with the authoritative
functions in `math_ops.py`; stored dataset answers are not used as probe labels.
Because many property magnitudes change with permutation length, each target is
centered and scaled separately within every length using probe-training data.
The primary `length_conditioned_r2` therefore measures information beyond a
length-only conditional-mean baseline.

## Probe fitting and held-out evaluation

- 4,096 deterministic prefixes from the validation split are used for probe
  fitting and regularization selection.
- Within every permutation length, 75% are assigned to probe training and 25%
  to tuning by a frozen hash split.
- A linear ridge probe is fitted independently at the embedding output, every
  Transformer block, and final normalization layer.
- The ridge coefficient is selected separately for each property from the
  frozen grid using tuning `length_conditioned_r2`.
- The selected probe is refitted on all 4,096 validation examples.
- It is evaluated once on 4,096 independently selected examples from the
  property test split.

In addition to length-conditioned R2, the report includes Pearson correlation,
normalized RMSE, rounded exact-property accuracy, and the corresponding
length-conditional modal-label baseline. Negative R2 means the linear probe is
worse than predicting the training conditional mean for each length.

## Statistical units

The primary target set for each model is the 16 properties in the opposite
pool. Metrics are first macro-averaged across those properties within one
model. Pool A and Pool B directions are then averaged within each replicate.
Finally, the report gives the mean and sample standard deviation across the
three joint task-split/model-seed replicates.

This produces one replicate-level value per `k`, rather than treating 32
properties, two pool directions, and multiple layers as independent
replicates. Trained and same-pool-untrained probes are retained as diagnostics.
Three random-initialization Transformers provide a representation baseline.

The primary endpoint is final-layer opposite-pool
`length_conditioned_r2` across `k = 1, 2, 4, 8, 16`. Layerwise values and exact
accuracy are secondary. The analysis does not assume that a larger task count
must produce a monotonic trend.

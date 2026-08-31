# Henry permutation training: completion report

Generated from the frozen completion markers in `runs/henry-permutation` on
2026-08-30. The checkpoint directory is intentionally gitignored.

## Completion and integrity

- Formal matrix: **30/30 completed** (`2 architectures x 5 task counts x 3 seeds`).
- Architectures: 15 causal Transformers and 15 causal token-mixing MLPs.
- Task counts: 1, 2, 4, 8, and 16; seeds: 17, 42, and 314159.
- Every run reached 20,000 optimizer steps under the frozen protocol.
- Total training exposure: 38,399,232 examples and 807,897,938 supervised
  target tokens (answer tokens plus EOS).
- Strict audit: 30 passed, 0 incomplete, 0 failed. The audit checked the full
  launch configuration, data split fingerprints, checkpoint SHA-256, strict
  model state shape/dtype compatibility, optimizer/scheduler/scaler/RNG state,
  marker/checkpoint accounting agreement, finite values, and partial files.
- Repository test suite: all 147 tests passed.

The formal configuration SHA-256 is
`c5d9a0ea7a601588d1e07a520721dfeb3b8f96830d03c8c9f8632c6d37f70dfa`.

## Data

- Schema: `permutation-20/v2`.
- Parent corpus: 10,000,000 records in 100 gzip shards (1.29 GB), with exactly
  500,000 examples for each of the 20 permutation tasks.
- Split: 9,800,000 train records (shards 000-097), 100,000 validation records
  (shard 098), and 100,000 test records (shard 099).
- Full verification independently recomputed all 20 mathematical answers and
  checked every canonical Passage Math token sequence for all 10,000,000 rows.
- Parent manifest SHA-256:
  `a9cc873bc82777c50fc2cfced96f54d727e3c3964eff457bd1a03ffabb179e87`.

## Frozen model and optimizer protocol

Both models use a 163-token vocabulary, 1,024-token context, tied embeddings,
`d_model=256`, dropout 0.1, and feed-forward ratio 4. The Transformer has four
layers and eight heads (3,463,424 parameters); the causal MLP has one layer
(2,930,176 parameters). Every run uses AdamW, learning rate 3e-4, weight decay
0.01, 1,000 warmup steps, cosine decay to 0.1 of the initial rate, bf16 AMP,
micro-batch 16, four-step gradient accumulation, and a 4,096-token dynamic
batch budget.

## Final validation snapshot

Values are mean +/- sample standard deviation across the three seeds. “Seen”
is the macro-average over that run's training tasks. “Pool-unseen” covers tasks
among the nested 16-task pool that were not trained at that task count.
“Holdout-4” is the fixed global holdout set: `to_reduced_word`, `compose`,
`parity`, and `to_lehmer`.

| Architecture | Tasks | Seen token % | Seen seq % | Pool-unseen token % | Pool-unseen seq % | Holdout-4 token % | Holdout-4 seq % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1 | 74.72 +/- 0.10 | 25.43 +/- 1.96 | 23.25 +/- 0.51 | 0.29 +/- 0.07 | 19.18 +/- 1.02 | 0.23 +/- 0.00 |
| Transformer | 2 | 81.18 +/- 0.60 | 52.80 +/- 1.73 | 34.41 +/- 2.21 | 2.73 +/- 0.49 | 25.71 +/- 1.36 | 0.88 +/- 0.58 |
| Transformer | 4 | 84.82 +/- 1.34 | 56.44 +/- 2.82 | 30.36 +/- 1.19 | 0.33 +/- 0.19 | 30.89 +/- 0.29 | 0.00 +/- 0.00 |
| Transformer | 8 | 83.33 +/- 0.28 | 50.80 +/- 0.55 | 37.88 +/- 1.37 | 0.23 +/- 0.00 | 43.75 +/- 0.22 | 0.39 +/- 0.13 |
| Transformer | 16 | 77.93 +/- 0.23 | 38.72 +/- 0.25 | N/A | N/A | 41.03 +/- 1.94 | 0.31 +/- 0.13 |
| MLP | 1 | 69.44 +/- 0.15 | 17.09 +/- 0.98 | 21.40 +/- 0.33 | 0.00 +/- 0.00 | 23.08 +/- 0.38 | 0.00 +/- 0.00 |
| MLP | 2 | 72.94 +/- 0.39 | 35.19 +/- 0.83 | 37.36 +/- 2.22 | 3.91 +/- 0.10 | 35.14 +/- 2.71 | 1.17 +/- 0.13 |
| MLP | 4 | 74.11 +/- 0.42 | 39.36 +/- 0.74 | 34.40 +/- 0.85 | 0.89 +/- 0.42 | 40.99 +/- 2.75 | 3.12 +/- 1.80 |
| MLP | 8 | 75.16 +/- 0.27 | 36.40 +/- 0.70 | 41.24 +/- 0.85 | 0.93 +/- 0.36 | 49.05 +/- 0.78 | 1.06 +/- 0.53 |
| MLP | 16 | 70.03 +/- 0.62 | 27.99 +/- 0.76 | N/A | N/A | 43.80 +/- 0.56 | 0.83 +/- 1.04 |

The Transformer-minus-MLP paired-seed difference in seen-task sequence
accuracy is +8.33, +17.61, +17.08, +14.40, and +10.73 percentage points for
1, 2, 4, 8, and 16 tasks. The corresponding difference in fixed-holdout token
accuracy is -3.90, -9.43, -10.10, -5.31, and -2.77 points.

## Preliminary zero-shot generalization result

The fixed four-task holdout is the clean comparison because those tasks were
excluded from training for every model. Holdout token accuracy rises from
19.18% to 43.75% for
the Transformer and from 23.08% to 49.05% for the MLP between the one-task and
eight-task conditions, then falls modestly at 16 tasks. This is evidence of
better prefix-conditioned token transfer with greater task diversity, with a
peak at eight tasks under the fixed optimizer-step budget.

Exact canonical-sequence generalization remains weak. Averaged over the four
holdouts, it never exceeds 0.88% for the Transformer or 3.12% for the MLP.
`to_reduced_word` and `to_lehmer` have 0% exact accuracy in every condition.
The nonzero results come from `compose` (best three-seed mean: 1.54% for the
Transformer and 2.16% for the MLP) and `parity` (best: 2.92% and 12.50%).

The four held-out operations use opaque task tokens that never occur during
base-model training. Consequently, hard zero-shot task identification is
underdetermined: the model is never taught what those four tokens mean. The
more informative Henry comparisons are therefore few-shot adaptation against
a random-initialization baseline and linear probing of pre-answer hidden
states; neither has been run yet.

## Interpretation boundary

These are validation diagnostics from shard 098, which was consulted throughout
training. Token accuracy uses teacher forcing, so every answer token is
predicted with the gold answer prefix available and copy/formatting tokens can
inflate it. `sequence_accuracy` requires every answer token and EOS argmax to
be correct. Because the models are strictly causal, that all-token event is
equivalent to greedy exact generation of the canonical target for the same
prompt, although a separate decoding harness has not yet been run.

Shard 099 remains untouched by model evaluation. A frozen test pass, few-shot
holdout fine-tuning, random-initialization baselines, linear probes, and
representation-geometry analysis remain downstream experiments. Accordingly,
this report establishes completion of all 30 base-model runs and a preliminary
validation-set zero-shot result; it is not the completed Henry generalization
study.

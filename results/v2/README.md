# v2 baseline results

V2 is the original nested-task baseline. It is kept separate from the revised
[v3 experiment](../v3/README.md) and must not be relabeled as a v3 result.

## Completion

- 30/30 models completed: 2 architectures x 5 task counts x 3 seeds.
- Every model reached 20,000 optimizer updates.
- Strict audit: 30 passed, 0 incomplete, 0 failed.
- Total training exposure: 38,399,232 examples and 807,897,938 supervised
  target tokens, including EOS.
- Frozen configuration SHA-256:
  `c5d9a0ea7a601588d1e07a520721dfeb3b8f96830d03c8c9f8632c6d37f70dfa`.

The v2 corpus contains 10,000,000 records in 100 gzip shards, exactly 500,000
examples for each of 20 tasks. It uses 9.8M training, 100k validation, and 100k
test records. All records passed answer recomputation and canonical Passage
Math verification. Parent manifest SHA-256:
`a9cc873bc82777c50fc2cfced96f54d727e3c3964eff457bd1a03ffabb179e87`.

## Architecture and training

Both models use a 163-token vocabulary, a 1,024-token context, tied
embeddings, `d_model=256`, dropout 0.1, and feed-forward ratio 4.

| Architecture | Configuration | Parameters |
|---|---|---:|
| Transformer | Standard pre-LN causal decoder, 4 layers, 8 heads | 3,463,424 |
| MLP | One causal token-mixing layer and a channel MLP | 2,930,176 |

Every run uses AdamW, learning rate `3e-4`, weight decay 0.01, 1,000 warmup
steps, cosine decay, bfloat16 AMP, micro-batch 16, and four-step gradient
accumulation.

## Validation results

The first `k` tasks of one frozen 16-task order were used for training. The
remaining pool tasks are `pool_unseen`; `to_reduced_word`, `compose`, `parity`,
and `to_lehmer` form the fixed four-task training holdout. Values are task
macro mean +/- sample standard deviation across three seeds.

| Architecture | Trained tasks | Seen token % | Seen exact % | Pool-unseen token % | Pool-unseen exact % | Holdout-4 token % | Holdout-4 exact % |
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

Token transfer improves with task diversity through the eight-task condition,
but exact holdout accuracy remains weak and non-monotonic. These are
validation diagnostics from shard 098, which was consulted during training;
v2 test shard 099 was not evaluated. The completed v3 study provides the
independent-test result used for the main paper conclusion.

## Raw results

[`model_task_accuracies.csv`](model_task_accuracies.csv) contains all 600
unaveraged model-task rows: 30 models x 20 tasks. Each row identifies the run,
architecture, trained task count, seed, evaluated task, task status, validation
example/token counts, teacher-forced token accuracy, and exact sequence
accuracy.

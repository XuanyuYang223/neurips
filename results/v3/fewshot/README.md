# Henry-style 20-shot adaptation results

This follow-up implements Henry Kvinge's proposed fine-tuning notion of
generalization: adapt each nested v3 model to an unseen holdout using 20
labeled examples and a low learning rate, then compare with a randomly
initialized model trained on the same support set.

## Main conclusion

Greater base-task diversity made the Transformer substantially easier to
adapt in loss, token accuracy, and the short Boolean `parity` task. It did not
produce reliable 20-shot execution of the three structured operations.

- Transformer four-task exact accuracy rose from 3.37% at one base task to
  12.68% at eight tasks and 12.49% at sixteen tasks.
- MLP exact accuracy rose from 1.72% to a non-monotonic maximum of 9.38% at
  four tasks.
- Nearly all of this exact accuracy came from `parity`. Across
  `to_reduced_word`, `compose`, and `to_lehmer`, pretrained Transformer exact
  accuracy peaked at 0.113%, and pretrained MLP accuracy was always 0%.
- The generous random-init controls, which used the established from-scratch
  learning rate, reached 13.31%/13.00% four-task exact accuracy for the
  Transformer/MLP and 1.216%/0.896% on the three structured tasks.

Thus the 20-shot result provides evidence for progressively better
prefix-conditioned adaptation in the Transformer, but not for broad
few-shot acquisition of unseen permutation algorithms.

## Completion and integrity

| Item | Result |
|---|---:|
| Pretrained adaptations | 120/120 completed |
| Random-init controls | 24/24 completed |
| Strict checkpoint audit | 144 passed, 0 failed |
| Support examples per run | 20 |
| Training presentations per run | 800 |
| Optimizer steps per run | 200 |
| Validation examples per run | 5,000 |
| Test examples per run | 5,000 |
| Total adaptation-test examples | 720,000 |

All support examples came from train shards 000-097. The 144 checkpoints were
audited for identity, hashes, model structure, finite tensors and metrics,
final learning rate, exposure count, and full target-task validation count
before test shard 099 was read. Test results are bound to implementation commit
`8d3ec126fb9cb88a1640a0d2d3497f0c93275c2d`, configuration SHA-256
`8151ca27cabc61db753714bc3003db88815553a3ad1220e6f54268bc72714ad0`,
support SHA-256
`a9463d6402dde0425048ff156f94a32e6d2115cc4cbed3788398333ae3aedc0a`,
and the frozen test-manifest SHA-256
`3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b`.

## Four-holdout average

Values macro-average the four holdouts within each seed and report mean +/-
sample standard deviation over three paired seeds.

| Initialization | Architecture | Base tasks | Loss | Token % | Exact % |
|---|---|---:|---:|---:|---:|
| Pretrained | Transformer | 1 | 7.132 +/- 0.512 | 16.17 +/- 0.42 | 3.37 +/- 1.25 |
| Pretrained | Transformer | 2 | 5.734 +/- 0.227 | 19.05 +/- 0.38 | 7.04 +/- 0.53 |
| Pretrained | Transformer | 4 | 2.155 +/- 0.076 | 56.21 +/- 0.37 | 11.01 +/- 0.26 |
| Pretrained | Transformer | 8 | 1.542 +/- 0.018 | 59.18 +/- 0.14 | 12.68 +/- 0.32 |
| Pretrained | Transformer | 16 | 1.481 +/- 0.071 | 59.14 +/- 0.47 | 12.49 +/- 0.56 |
| Random init | Transformer | 0 | 1.712 +/- 0.020 | 60.35 +/- 0.21 | 13.31 +/- 0.37 |
| Pretrained | MLP | 1 | 8.430 +/- 0.348 | 14.85 +/- 0.14 | 1.72 +/- 0.01 |
| Pretrained | MLP | 2 | 7.671 +/- 0.156 | 14.88 +/- 0.14 | 1.61 +/- 0.17 |
| Pretrained | MLP | 4 | 4.587 +/- 0.575 | 43.46 +/- 2.14 | 9.38 +/- 1.44 |
| Pretrained | MLP | 8 | 5.472 +/- 0.279 | 41.09 +/- 0.77 | 5.08 +/- 1.42 |
| Pretrained | MLP | 16 | 3.537 +/- 0.806 | 44.49 +/- 2.31 | 5.50 +/- 4.08 |
| Random init | MLP | 0 | 1.865 +/- 0.023 | 59.40 +/- 0.20 | 13.00 +/- 0.48 |

Compared with each model's frozen zero-shot result, Transformer exact accuracy
improved by +2.00, +5.40, +9.46, +11.88, and +12.20 percentage points at
1, 2, 4, 8, and 16 base tasks. The MLP changes were +0.01, +0.22, +6.55,
+5.07, and +4.55 points. This supports a progressive adaptation effect for
the Transformer through eight tasks, but not a monotonic MLP effect.

The random controls used `3e-4`, while pretrained adaptation used Henry's low
`1e-5` learning rate. This intentionally gives scratch training a reasonable
optimizer, but means the pretrained-versus-random difference is not an
identical-learning-rate ablation. The paired progression across pretrained
task counts remains learning-rate matched.

## Structured holdouts only

Because `parity` has a two-token Boolean target, it can dominate a four-task
exact average. The following table removes `parity` and macro-averages only
`to_reduced_word`, `compose`, and `to_lehmer`.

| Initialization | Architecture | Base tasks | Loss | Token % | Exact % |
|---|---|---:|---:|---:|---:|
| Pretrained | Transformer | 1 | 8.508 +/- 0.897 | 2.65 +/- 0.40 | 0.000 +/- 0.000 |
| Pretrained | Transformer | 2 | 6.992 +/- 0.396 | 4.15 +/- 0.25 | 0.000 +/- 0.000 |
| Pretrained | Transformer | 4 | 2.522 +/- 0.050 | 50.94 +/- 0.35 | 0.000 +/- 0.000 |
| Pretrained | Transformer | 8 | 1.843 +/- 0.026 | 53.85 +/- 0.26 | 0.113 +/- 0.133 |
| Pretrained | Transformer | 16 | 1.698 +/- 0.017 | 53.90 +/- 0.31 | 0.078 +/- 0.107 |
| Random init | Transformer | 0 | 1.850 +/- 0.022 | 55.54 +/- 0.24 | 1.216 +/- 0.294 |
| Pretrained | MLP | 1 | 9.176 +/- 0.326 | 1.99 +/- 0.20 | 0.000 +/- 0.000 |
| Pretrained | MLP | 2 | 8.504 +/- 0.204 | 2.10 +/- 0.12 | 0.000 +/- 0.000 |
| Pretrained | MLP | 4 | 5.617 +/- 0.624 | 35.03 +/- 1.99 | 0.000 +/- 0.000 |
| Pretrained | MLP | 8 | 6.707 +/- 0.386 | 34.73 +/- 0.93 | 0.000 +/- 0.000 |
| Pretrained | MLP | 16 | 4.163 +/- 1.099 | 38.98 +/- 2.83 | 0.000 +/- 0.000 |
| Random init | MLP | 0 | 2.091 +/- 0.017 | 54.32 +/- 0.15 | 0.896 +/- 0.401 |

Loss and token accuracy improve strongly with broader Transformer pretraining,
but complete structured answers remain almost entirely incorrect. This is the
most important qualification on the apparent four-task adaptation curve.

## Per-task exact accuracy

Three-seed mean exact accuracy is shown below.

| Architecture | Base tasks | Reduced word % | Compose % | Parity % | Lehmer % |
|---|---:|---:|---:|---:|---:|
| Transformer | 1 | 0.00 | 0.00 | 13.49 | 0.00 |
| Transformer | 2 | 0.00 | 0.00 | 28.14 | 0.00 |
| Transformer | 4 | 0.00 | 0.00 | 44.05 | 0.00 |
| Transformer | 8 | 0.00 | 0.34 | 50.39 | 0.00 |
| Transformer | 16 | 0.00 | 0.00 | 49.74 | 0.23 |
| Transformer random | 0 | 1.27 | 0.87 | 49.61 | 1.51 |
| MLP | 1 | 0.00 | 0.00 | 6.89 | 0.00 |
| MLP | 2 | 0.00 | 0.00 | 6.43 | 0.00 |
| MLP | 4 | 0.00 | 0.00 | 37.51 | 0.00 |
| MLP | 8 | 0.00 | 0.00 | 20.30 | 0.00 |
| MLP | 16 | 0.00 | 0.00 | 22.01 | 0.00 |
| MLP random | 0 | 0.79 | 0.59 | 49.29 | 1.31 |

## Result files

- [`test_model_task_accuracies.csv`](test_model_task_accuracies.csv): all 144
  unaveraged adaptation-task test rows;
- [`test_summary.csv`](test_summary.csv): four-holdout macro results;
- [`test_structured_summary.csv`](test_structured_summary.csv): the same
  aggregation with Boolean `parity` excluded;
- [`test_task_summary.csv`](test_task_summary.csv): every task and condition,
  averaged only across seeds;
- [`test_adaptation_gains.csv`](test_adaptation_gains.csv): paired improvement
  over zero-shot and random initialization;
- [`evaluation/manifest.json`](evaluation/manifest.json): frozen test
  evaluation identity and per-run index;
- [the preregistered protocol](../FEW_SHOT_PROTOCOL.md): support selection,
  optimization, comparison, and leakage controls.

Raw accuracies are fractions in the CSVs and percentages in this report.
`token_accuracy` is teacher-forced; `sequence_accuracy` is the primary complete
answer metric.

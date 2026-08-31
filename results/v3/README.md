# v3 revised experiment results

This package reports the completed experiment after Henry Kvinge's feedback.
It contains 48 trained models, their final validation metrics, and the single
frozen evaluation on independent test shard 099. The older experiment is kept
separately in the [v2 result package](../v2/README.md).

## Main conclusion

All 48 models completed 20,000 optimizer updates and passed strict audit.
Training-task accuracy was often high, but the models did not reliably execute
operations that were excluded from their training data.

The primary generalization averages below therefore **exclude all seen
training tasks**. This prevents successful memorization or learning of trained
operations from inflating the generalization result.

- On the four fixed nested holdouts, increasing task diversity substantially
  reduced loss and raised teacher-forced token accuracy, but exact complete
  answers remained between 0.00% and 2.83%.
- In the category comparison, exact accuracy on the eight matched tasks from
  the other two families was 0.00%-2.50%, despite strong same-family learning.
- `to_reduced_word`, `compose`, and `to_lehmer` remained at 0% exact accuracy
  in every nested condition. Small nonzero holdout averages came from the
  short Boolean `parity` output.

The defensible result is: **more varied training can transfer output-format
and local token regularities, but this experiment does not show reliable
zero-shot acquisition of an unseen permutation operation.**

## Generalization-only nested results

The fixed holdout set is identical for every nested model:
`to_reduced_word`, `compose`, `parity`, and `to_lehmer`. None of these tasks
contributed training examples or gradient targets. Values are task-macro
mean +/- sample standard deviation over three seeds. Changes are relative to
the one-task model of the same architecture; lower loss and positive accuracy
changes are better.

| Architecture | Trained tasks | Loss | Loss change | Token % | Token change (pp) | Exact % | Exact change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1 | 12.799 +/- 0.421 | +0.000 | 14.62 +/- 0.97 | +0.00 | 1.37 +/- 0.19 | +0.00 |
| Transformer | 2 | 12.576 +/- 0.620 | -0.223 | 14.75 +/- 1.12 | +0.12 | 1.63 +/- 0.30 | +0.26 |
| Transformer | 4 | 6.355 +/- 0.542 | -6.444 | 32.89 +/- 2.64 | +18.27 | 1.55 +/- 0.18 | +0.18 |
| Transformer | 8 | 6.315 +/- 0.725 | -6.484 | 29.87 +/- 0.93 | +15.25 | 0.80 +/- 0.94 | -0.57 |
| Transformer | 16 | 5.907 +/- 0.732 | -6.892 | 29.05 +/- 3.57 | +14.43 | 0.30 +/- 0.31 | -1.07 |
| MLP | 1 | 11.244 +/- 0.544 | +0.000 | 14.44 +/- 0.40 | +0.00 | 1.71 +/- 0.00 | +0.00 |
| MLP | 2 | 11.320 +/- 0.939 | +0.076 | 14.11 +/- 0.39 | -0.34 | 1.39 +/- 0.11 | -0.32 |
| MLP | 4 | 6.945 +/- 0.698 | -4.299 | 36.63 +/- 1.75 | +22.18 | 2.83 +/- 1.14 | +1.11 |
| MLP | 8 | 9.049 +/- 0.241 | -2.195 | 34.26 +/- 0.73 | +19.82 | 0.00 +/- 0.00 | -1.71 |
| MLP | 16 | 7.101 +/- 0.807 | -4.143 | 33.17 +/- 0.47 | +18.72 | 0.95 +/- 0.25 | -0.76 |

Loss and token accuracy improve markedly after four or more training tasks,
but exact answer accuracy does not improve monotonically. The loss reduction
therefore reflects better probability assignment along the gold answer path,
not reliable completion of the unseen operation.

The remaining tasks in the 16-task training pool provide a second unseen-task
diagnostic. This set shrinks from 15 tasks at `k=1` to 8 tasks at `k=8`, so
changes are less directly comparable than the fixed-holdout table.

| Architecture | Trained tasks | Unseen tasks | Loss | Loss change | Token % | Token change (pp) | Exact % | Exact change (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1 | 15 | 9.389 +/- 0.225 | +0.000 | 31.30 +/- 0.60 | +0.00 | 7.51 +/- 0.38 | +0.00 |
| Transformer | 2 | 14 | 9.933 +/- 0.435 | +0.544 | 30.07 +/- 0.57 | -1.23 | 7.30 +/- 0.64 | -0.21 |
| Transformer | 4 | 12 | 6.387 +/- 0.181 | -3.001 | 37.30 +/- 2.37 | +6.00 | 7.52 +/- 0.63 | +0.01 |
| Transformer | 8 | 8 | 6.033 +/- 0.424 | -3.356 | 37.77 +/- 1.30 | +6.47 | 2.77 +/- 1.39 | -4.74 |
| MLP | 1 | 15 | 9.188 +/- 0.335 | +0.000 | 29.93 +/- 0.84 | +0.00 | 8.38 +/- 0.01 | +0.00 |
| MLP | 2 | 14 | 9.444 +/- 0.294 | +0.256 | 27.00 +/- 0.58 | -2.93 | 6.83 +/- 0.12 | -1.55 |
| MLP | 4 | 12 | 5.952 +/- 0.354 | -3.236 | 40.93 +/- 0.11 | +11.00 | 4.25 +/- 2.17 | -4.12 |
| MLP | 8 | 8 | 6.580 +/- 0.496 | -2.608 | 42.72 +/- 1.28 | +12.78 | 1.82 +/- 0.83 | -6.56 |

The exact machine-readable values are in
[`test_nested_generalization.csv`](test_nested_generalization.csv).

## Generalization-only category results

The category study trains on four tasks from one family and evaluates four
matched tasks from each of the other two families. The table below averages
only those eight off-diagonal tasks; the four seen tasks are excluded. The gap
is unseen minus seen for the same run, so a positive loss gap or a negative
accuracy gap indicates worse generalization.

| Architecture | Training family | Unseen loss | Loss gap | Unseen token % | Token gap (pp) | Unseen exact % | Exact gap (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| Transformer | Encoding E4 | 9.499 +/- 0.047 | +9.483 | 28.73 +/- 2.15 | -70.80 | 0.00 +/- 0.00 | -84.45 |
| Transformer | Statistics S4 | 10.481 +/- 0.434 | +10.146 | 17.00 +/- 0.54 | -69.91 | 2.50 +/- 0.17 | -45.42 |
| Transformer | Algebra A4 | 9.157 +/- 0.747 | +9.153 | 31.09 +/- 1.44 | -68.80 | 0.02 +/- 0.03 | -97.59 |
| MLP | Encoding E4 | 7.874 +/- 0.509 | +7.218 | 27.32 +/- 3.21 | -48.76 | 0.00 +/- 0.00 | -16.23 |
| MLP | Statistics S4 | 9.768 +/- 0.452 | +9.312 | 24.02 +/- 0.83 | -57.62 | 2.50 +/- 1.00 | -35.36 |
| MLP | Algebra A4 | 8.745 +/- 0.432 | +8.145 | 30.45 +/- 1.49 | -48.97 | 0.15 +/- 0.14 | -30.70 |

The nonzero Statistics-to-other-family exact average is driven by transfer
between short Boolean outputs, especially `pattern_avoidance` and
`bruhat_leq`; it is not broad success on permutation-output operations. Exact
values are in [`test_category_generalization.csv`](test_category_generalization.csv).

## Dataset, models, and training

Henry suggested removing the slow `power`, `conjugate`, and `commutator`
tasks and comparing representation families. V3 replaced them with `peaks`,
`exceedances`, and `recoils`, each with 500,000 records. The final corpus has
20 tasks x 500,000 records = 10,000,000 records, split into 9.8M training,
100k validation, and 100k test examples. Permutation sizes are `2 <= n <= 30`.

Both architectures use a 163-token vocabulary, 1,024-token context,
`d_model=256`, dropout 0.1, and tied embeddings.

| Architecture | Structure | Registered parameters |
|---|---|---:|
| Transformer | Standard pre-LN causal decoder, 4 layers, 8 heads, FFN width 1,024 | 3,463,424 |
| MLP | One causal token-mixing block and a position-wise channel MLP | 2,930,176 |

Every model was trained from scratch for 20,000 optimizer steps with AdamW,
learning rate `3e-4`, weight decay 0.01, 1,000 warmup steps, cosine decay,
gradient clipping at 1.0, and bfloat16 AMP. The experiment used seeds `17`,
`42`, and `314159`.

The nested matrix is 2 architectures x 5 task counts x 3 seeds = 30 models.
The category matrix is 2 architectures x 3 task families x 3 seeds = 18
models. Category E4 used micro-batch 4 with 16 accumulation steps; all other
conditions used micro-batch 16 with four accumulation steps. Every optimizer
update therefore represented 64 examples.

The complete launch and integrity record is in [LAUNCH.md](LAUNCH.md). Exact
encoding and mathematical definitions are in [the protocol](../../PROTOCOL.md),
and the full operational history is in
[the training process](../../TRAINING_PROCESS.md).

## Metric interpretation

- `loss` is negative log likelihood over supervised answer tokens and EOS,
  macro-averaged equally across tasks. Lower is better.
- `token_accuracy` is teacher-forced: every answer position sees the correct
  preceding answer tokens. It includes delimiters, copied tokens, and EOS.
- `sequence_accuracy` requires every answer token and EOS argmax to be correct.
  It is the primary complete-answer metric.

The test split is independent at the example level but follows the same
`n=2..30` distribution. These results do not establish size extrapolation,
cross-representation input transfer, or learned-representation similarity.
CKA, linear probing, and few-shot adaptation remain separate future analyses.

## Integrity and provenance

| Item | Result |
|---|---:|
| Nested strict audit | 30/30 passed |
| Category strict audit | 18/18 passed |
| Failed or incomplete runs | 0 |
| Test examples per model | 100,000 |
| Test examples per task per model | 5,000 |
| Total model-test examples | 4,800,000 |
| Test evaluation failures | 0 |

Frozen identifiers:

- training implementation commit: `d1d163bf2b3209dc5b6cc61ac4396d84fa6e2613`;
- launch commit: `c346cb92ae46f88b01912e4ff3c1dc7b22e4b9a8`;
- evaluator commit: `7e2933033f6d99e16a004430d3dce19d51f37013`;
- experiment configuration SHA-256:
  `dd75f31277e42f554ed681beda44bb53f2d4f65089fd9583540b9e645c4f1b40`;
- v3 parent manifest SHA-256:
  `b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f`;
- test manifest SHA-256:
  `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b`.

## Result files

The redundant `protocol_version` column was removed from public v3 CSVs; the
version remains unambiguous from this directory and all provenance hashes are
retained.

- [`test_model_task_accuracies.csv`](test_model_task_accuracies.csv): 960 raw
  independent-test model-task rows;
- [`test_nested_generalization.csv`](test_nested_generalization.csv): nested
  unseen-task loss and accuracy changes, excluding seen tasks;
- [`test_category_generalization.csv`](test_category_generalization.csv):
  off-diagonal category generalization, excluding seen tasks;
- [`test_run_summaries.csv`](test_run_summaries.csv): per-run task-group macros;
- [`test_nested_summary.csv`](test_nested_summary.csv): all nested groups,
  including seen-task diagnostics;
- [`test_category_summary.csv`](test_category_summary.csv): complete 3x3
  category matrix, including the seen diagonal;
- [`validation_model_task_accuracies.csv`](validation_model_task_accuracies.csv):
  960 validation rows;
- [`evaluation/manifest.json`](evaluation/manifest.json): frozen test provenance
  and per-run result index.

Regenerate every CSV from authenticated local artifacts with:

```bash
permutation-results \
  --config configs/henry_permutation_revised.toml \
  --output-dir results/v3 \
  --test-evaluation-dir results/v3/evaluation
```

Run this command from the repository root. It strictly audits all models before
writing any aggregate table.

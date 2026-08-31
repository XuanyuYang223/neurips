# v3 Training and Generalization Results

This document reports the completed revised experiment after Henry Kvinge's
feedback. It covers all 48 v3 models, the final validation snapshot, and the
single frozen evaluation on the previously unused test shard. The older
[`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) remains the separate v2 baseline.

## Main result

All **48/48** planned v3 models completed 20,000 optimizer updates and passed
strict checkpoint audit. On the independent test split, models learned their
trained tasks well, but did **not** reliably execute unseen permutation tasks.

- In the nested study, exact accuracy on the four tasks held out from every
  gradient update remained between 0% and 2.83% on average, with no monotonic
  improvement as the number of training tasks increased.
- `to_reduced_word`, `compose`, and `to_lehmer` had exactly 0% test exact
  accuracy in every nested condition and architecture. All nonzero holdout
  exact accuracy came from the two-token Boolean `parity` target.
- In the category study, off-diagonal exact transfer was essentially zero.
  The apparent 4.99%/5.00% Statistics-to-Algebra result came entirely from
  approximately 20% accuracy on the Boolean `bruhat_leq` task; the three
  permutation-output algebra tasks remained at 0% exact accuracy.
- Same-category learning was strong for the Transformer: 84.45% exact for E4,
  47.91% for S4, and 97.61% for A4. The MLP reached 16.23%, 37.85%, and 30.86%.

The defensible conclusion is therefore: **the models learned trained
operations, output formats, and some local token regularities, but these
experiments do not show reliable zero-shot acquisition of an untrained
permutation operation.**

## Completion and integrity

| Item | Result |
|---|---:|
| Nested matrix | 30/30 passed strict audit |
| Category matrix | 18/18 passed strict audit |
| Failed or incomplete runs | 0 |
| Optimizer updates per run | 20,000 |
| Total training examples | 61,439,232 |
| Total supervised target tokens, including EOS | 1,048,870,420 |
| Checkpoint bytes | 1,844,025,024 |
| Independent test examples per model | 100,000 |
| Independent test examples per task per model | 5,000 |
| Total model-test examples | 4,800,000 |
| Test forward-pass time, summed over models | 771.10 seconds |
| Test evaluation failures | 0 |

The strict audits checked the complete expected `TrainConfig`, dataset and
validation fingerprints, completion-marker/checkpoint agreement, checkpoint
SHA-256, model and optimizer structure, scheduler/scaler/RNG state, finite
tensors, task accounting, and all 20 validation entries. Both audits reported
zero global issues, partial artifacts, or forbidden symlinks.

Frozen provenance:

- training implementation commit:
  `d1d163bf2b3209dc5b6cc61ac4396d84fa6e2613`;
- formal launch commit:
  `c346cb92ae46f88b01912e4ff3c1dc7b22e4b9a8`;
- one-time test evaluator commit:
  `7e2933033f6d99e16a004430d3dce19d51f37013`;
- experiment configuration SHA-256:
  `dd75f31277e42f554ed681beda44bb53f2d4f65089fd9583540b9e645c4f1b40`;
- v3 parent manifest SHA-256:
  `b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f`;
- test manifest SHA-256:
  `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b`.

The frozen TOML still contains launch-time strings such as
`planned_not_trained`. Those fields are intentionally unchanged because every
checkpoint authenticates the exact TOML bytes above. This result document and
the strict completion artifacts record the post-run status.

## Henry's revision and the 10M-record dataset

Henry recommended a standard architecture, suggested removing the unusually
slow `power`, `conjugate`, and `commutator` tasks, and proposed comparing
encoding, statistics, and algebra training conditions. The project selected
three linear-time scalar replacements:

| Removed v2 task | Added v3 property | Definition | Records |
|---|---|---|---:|
| `power` | `peaks` | Internal positions with `pi[i-1] < pi[i] > pi[i+1]` | 500,000 |
| `conjugate` | `exceedances` | Positions `i` with `pi(i) > i` | 500,000 |
| `commutator` | `recoils` | Descents of `pi^-1` | 500,000 |

The final dataset contains 20 tasks x 500,000 records = **10,000,000** records
in 100 gzip shards. It uses 9.8M training, 100k validation, and 100k test
records. All 10M records passed answer recomputation and canonical encoding
verification. Permutation sizes are sampled in `2 <= n <= 30`.

## Passage Math encoding

Each record is one canonical causal-language-model sequence:

```text
<BOS> <SIZE> n <ONE_START> pi(1) , ... , pi(n) <ONE_END>
<TASK_TOKEN> [typed operand] = <canonical answer> <EOS>
```

Numbers use the supplied base-100 convention. Values `0` through `99` are
single atomic tokens `00` through `99`; they deliberately do **not** use
`<NUM_START>`. Values at least 100 use
`<NUM_START> ...base-100 digits... <NUM_END>`. Since `peaks`, `exceedances`,
and `recoils` are at most 29 for `n <= 30`, each new answer is one number token.

An actual v3 example is:

```text
<BOS> <SIZE> 23 <ONE_START> 15 , 07 , ... , 10 <ONE_END>
<PEAKS> = 07 <EOS>
```

Binary operations and comparisons add `<ARG_START> ... <ARG_END>`; pattern
avoidance uses `<PATTERN_START> ... <PATTERN_END>`; simple multiplication uses
`<SIMPLE_INDEX>`. Structured answers use their task-specific boundaries.
Exact grammar and mathematical conventions are in [PROTOCOL.md](PROTOCOL.md).

## Architectures

The same 163-token vocabulary, 1,024-token context, `d_model=256`, dropout
0.1, feed-forward ratio 4, and tied input/output embeddings are used for both
architectures.

| Architecture | Structure | Registered parameters |
|---|---|---:|
| Transformer | Standard pre-LN decoder-only Transformer, 4 layers, 8 heads, hidden width 256, FFN width 1,024 | 3,463,424 |
| MLP | One strictly causal token-mixing MLP block plus a position-wise channel MLP, hidden width 256 | 2,930,176 |

The MLP count is nominal: its registered 1,024 x 1,024 token-mixing matrices
include upper-triangular entries that are masked out of the forward pass.
Architecture sizes are therefore similar, not exactly capacity-matched.

## Training protocol

Every model was initialized from scratch and trained with AdamW, learning rate
`3e-4`, weight decay 0.01, 1,000 warmup steps, cosine decay to 10% of the
initial learning rate, gradient clipping at 1.0, and bfloat16 AMP. Checkpoints
and validation diagnostics were written every 1,000 steps. Loss was averaged
within each example before averaging examples, preventing long reduced words
from dominating scalar targets.

Nested runs used micro-batch 16 with four accumulation steps. Category E4
used `4 x 16` because reduced words can be long; S4 and A4 used `16 x 4`.
Thus all formal conditions targeted 64 examples per optimizer update. The
30 nested runs consumed 38,399,232 examples; the 18 category runs consumed
exactly 23,040,000 examples.

The two formal matrices were:

- Nested: task counts 1, 2, 4, 8, and 16 x Transformer/MLP x three seeds = 30.
- Category: E4/S4/A4 x Transformer/MLP x three seeds = 18.

Seeds were `17`, `42`, and `314159`. Error bars below are the sample standard
deviation across these three seeds. Within each run, tasks are macro-averaged
before aggregation across seeds.

## Nested generalization: independent test results

“Seen” contains that run's training tasks. “Pool-unseen” contains the remaining
tasks in the 16-task nested pool. “Holdout-4” contains `to_reduced_word`,
`compose`, `parity`, and `to_lehmer`, which received no gradient updates in any
nested model. Values are task-macro mean +/- sample SD across seeds.

| Architecture | Training tasks | Seen token % | Seen exact % | Pool-unseen token % | Pool-unseen exact % | Holdout-4 token % | Holdout-4 exact % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer | 1 | 88.12 +/- 2.98 | 76.25 +/- 5.96 | 31.30 +/- 0.60 | 7.51 +/- 0.38 | 14.62 +/- 0.97 | 1.37 +/- 0.19 |
| Transformer | 2 | 94.87 +/- 0.61 | 89.75 +/- 1.23 | 30.07 +/- 0.57 | 7.30 +/- 0.64 | 14.75 +/- 1.12 | 1.63 +/- 0.30 |
| Transformer | 4 | 87.97 +/- 0.76 | 68.02 +/- 1.28 | 37.30 +/- 2.37 | 7.52 +/- 0.63 | 32.89 +/- 2.64 | 1.55 +/- 0.18 |
| Transformer | 8 | 86.93 +/- 0.52 | 59.60 +/- 0.33 | 37.77 +/- 1.30 | 2.77 +/- 1.39 | 29.87 +/- 0.93 | 0.80 +/- 0.94 |
| Transformer | 16 | 80.66 +/- 0.31 | 48.58 +/- 0.83 | N/A | N/A | 29.05 +/- 3.57 | 0.30 +/- 0.31 |
| MLP | 1 | 75.14 +/- 0.23 | 50.29 +/- 0.47 | 29.93 +/- 0.84 | 8.38 +/- 0.01 | 14.44 +/- 0.40 | 1.71 +/- 0.00 |
| MLP | 2 | 75.27 +/- 0.34 | 50.54 +/- 0.68 | 27.00 +/- 0.58 | 6.83 +/- 0.12 | 14.11 +/- 0.39 | 1.39 +/- 0.11 |
| MLP | 4 | 74.95 +/- 0.32 | 44.14 +/- 0.54 | 40.93 +/- 0.11 | 4.25 +/- 2.17 | 36.63 +/- 1.75 | 2.83 +/- 1.14 |
| MLP | 8 | 75.89 +/- 0.36 | 39.55 +/- 1.18 | 42.72 +/- 1.28 | 1.82 +/- 0.83 | 34.26 +/- 0.73 | 0.00 +/- 0.00 |
| MLP | 16 | 71.73 +/- 0.09 | 34.40 +/- 0.19 | N/A | N/A | 33.17 +/- 0.47 | 0.95 +/- 0.25 |

Token accuracy rises in some richer-task settings, but it is teacher-forced
and includes punctuation, boundaries, copied values, and EOS. The exact metric
does not show increasing generalization. The three structured holdout tasks
are 0% exact throughout; the small holdout averages above come only from
`parity`, where guessing one Boolean token and EOS can produce a complete hit.

## Category comparison: independent test results

The matched groups are:

- E4: `to_cycle`, `to_lehmer`, `to_inversion_vector`, `to_reduced_word`;
- S4: `length`, `cycle_type`, `rsk_shape`, `pattern_avoidance`;
- A4: `inverse`, `compose`, `right_multiply_simple`, `bruhat_leq`.

Each cell macro-averages the four evaluation tasks within a run and then
reports mean +/- sample SD across seeds.

| Architecture | Training group | Evaluation group | Token % | Exact % |
|---|---|---|---:|---:|
| Transformer | E4 | E4 | 99.53 +/- 0.22 | 84.45 +/- 4.86 |
| Transformer | E4 | S4 | 22.14 +/- 3.55 | 0.00 +/- 0.00 |
| Transformer | E4 | A4 | 35.32 +/- 1.09 | 0.00 +/- 0.00 |
| Transformer | S4 | E4 | 12.56 +/- 0.56 | 0.00 +/- 0.00 |
| Transformer | S4 | S4 | 86.91 +/- 0.79 | 47.91 +/- 0.72 |
| Transformer | S4 | A4 | 21.43 +/- 0.65 | 4.99 +/- 0.35 |
| Transformer | A4 | E4 | 34.97 +/- 0.46 | 0.00 +/- 0.00 |
| Transformer | A4 | S4 | 27.21 +/- 2.59 | 0.04 +/- 0.06 |
| Transformer | A4 | A4 | 99.89 +/- 0.09 | 97.61 +/- 1.61 |
| MLP | E4 | E4 | 76.08 +/- 0.65 | 16.23 +/- 0.91 |
| MLP | E4 | S4 | 25.09 +/- 3.18 | 0.00 +/- 0.00 |
| MLP | E4 | A4 | 29.55 +/- 3.38 | 0.00 +/- 0.00 |
| MLP | S4 | E4 | 21.12 +/- 1.58 | 0.00 +/- 0.00 |
| MLP | S4 | S4 | 81.64 +/- 0.29 | 37.85 +/- 0.25 |
| MLP | S4 | A4 | 26.91 +/- 0.23 | 5.00 +/- 1.99 |
| MLP | A4 | E4 | 40.90 +/- 2.41 | 0.00 +/- 0.00 |
| MLP | A4 | S4 | 19.99 +/- 0.60 | 0.30 +/- 0.28 |
| MLP | A4 | A4 | 79.42 +/- 0.62 | 30.86 +/- 0.67 |

The category diagonal confirms successful task learning, especially for the
Transformer. Off-diagonal exact accuracy does not support zero-shot operation
transfer. The S4-to-A4 nonzero cell is a target-type effect: `bruhat_leq` alone
reaches about 20% exact because S4 includes the Boolean `pattern_avoidance`
format, while `inverse`, `compose`, and `right_multiply_simple` remain 0%.

### Same-category task detail

| Architecture | Group | Task | Token % | Exact % |
|---|---|---|---:|---:|
| Transformer | E4 | `to_cycle` | 99.89 +/- 0.04 | 96.47 +/- 1.11 |
| Transformer | E4 | `to_inversion_vector` | 99.70 +/- 0.28 | 93.72 +/- 4.44 |
| Transformer | E4 | `to_lehmer` | 99.46 +/- 0.36 | 89.14 +/- 6.26 |
| Transformer | E4 | `to_reduced_word` | 99.09 +/- 0.32 | 58.45 +/- 9.11 |
| Transformer | S4 | `cycle_type` | 80.14 +/- 1.19 | 18.49 +/- 0.43 |
| Transformer | S4 | `length` | 82.97 +/- 0.63 | 48.70 +/- 2.01 |
| Transformer | S4 | `pattern_avoidance` | 97.81 +/- 0.11 | 95.63 +/- 0.22 |
| Transformer | S4 | `rsk_shape` | 86.70 +/- 1.76 | 28.84 +/- 1.44 |
| Transformer | A4 | `bruhat_leq` | 99.93 +/- 0.03 | 99.87 +/- 0.06 |
| Transformer | A4 | `compose` | 99.96 +/- 0.08 | 98.72 +/- 2.18 |
| Transformer | A4 | `inverse` | 99.75 +/- 0.40 | 94.48 +/- 8.40 |
| Transformer | A4 | `right_multiply_simple` | 99.92 +/- 0.09 | 97.38 +/- 3.10 |
| MLP | E4 | `to_cycle` | 70.73 +/- 0.82 | 14.71 +/- 0.75 |
| MLP | E4 | `to_inversion_vector` | 69.08 +/- 0.69 | 18.28 +/- 1.24 |
| MLP | E4 | `to_lehmer` | 71.89 +/- 1.47 | 17.04 +/- 1.26 |
| MLP | E4 | `to_reduced_word` | 92.63 +/- 0.11 | 14.87 +/- 0.66 |
| MLP | S4 | `cycle_type` | 71.30 +/- 0.76 | 11.01 +/- 0.55 |
| MLP | S4 | `length` | 77.64 +/- 0.08 | 33.53 +/- 0.34 |
| MLP | S4 | `pattern_avoidance` | 95.95 +/- 0.19 | 91.89 +/- 0.38 |
| MLP | S4 | `rsk_shape` | 81.67 +/- 0.18 | 14.99 +/- 0.45 |
| MLP | A4 | `bruhat_leq` | 97.84 +/- 0.30 | 95.68 +/- 0.59 |
| MLP | A4 | `compose` | 62.50 +/- 0.06 | 6.23 +/- 1.02 |
| MLP | A4 | `inverse` | 65.65 +/- 0.28 | 12.31 +/- 0.74 |
| MLP | A4 | `right_multiply_simple` | 91.69 +/- 2.39 | 9.21 +/- 0.75 |

## Metric interpretation

`token_accuracy` is teacher-forced: every later answer token is predicted with
the correct preceding answer tokens visible. It is useful diagnostically but
is not mathematical-answer accuracy. `sequence_accuracy` requires every
answer token and EOS argmax to be correct. For these strictly causal models,
that all-gold-path event is equivalent to deterministic greedy exact decoding
under the same tokenization.

The test split is independent at the example level but comes from the same
`n=2..30` distribution. These results do not test larger-size extrapolation,
cross-representation input transfer, or recovery after a decoding error.

The four nested holdout task tokens never occur as an input token or correct
target in nested base-model training, so they receive no operation-semantic
grounding, although their vocabulary rows still receive negative-class
softmax gradients. This makes direct zero-shot execution particularly hard
and motivates Henry's proposed few-shot and probing analyses.

The category accuracy matrix is a **behavioral transfer** analysis. It does
not itself measure learned representation similarity. Layerwise CKA, frozen
linear probes at `<ONE_END>`, and few-shot adaptation versus random
initialization remain future experiments and must not be described as done.

## Reproducible result files

- [`V3_TEST_MODEL_TASK_ACCURACIES.csv`](V3_TEST_MODEL_TASK_ACCURACIES.csv):
  all 960 independent test model-task cells;
- [`V3_TEST_RUN_SUMMARIES.csv`](V3_TEST_RUN_SUMMARIES.csv): per-run task-macro
  groups before seed averaging;
- [`V3_TEST_NESTED_SUMMARY.csv`](V3_TEST_NESTED_SUMMARY.csv): the 28 nested
  architecture/task-count/status summaries;
- [`V3_TEST_CATEGORY_SUMMARY.csv`](V3_TEST_CATEGORY_SUMMARY.csv): the 18
  category transfer cells;
- [`V3_MODEL_TASK_ACCURACIES.csv`](V3_MODEL_TASK_ACCURACIES.csv): all 960 final
  validation cells from shard098;
- [`evaluations/v3-test-shard099/manifest.json`](evaluations/v3-test-shard099/manifest.json):
  test provenance and per-run result index.

Regenerate the CSVs from authenticated checkpoints and evaluation files with:

```bash
permutation-results \
  --config configs/henry_permutation_revised.toml \
  --output-dir . \
  --test-evaluation-dir evaluations/v3-test-shard099
```

The test evaluator is resumable and identity-bound, but the reported test pass
was performed once after all model training and validation analysis were
frozen. No model selection or hyperparameter change was made after observing
shard099.

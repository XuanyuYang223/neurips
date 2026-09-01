# Zero-overlap 32-property study

## Three-replicate confirmatory extension

The primary result now averages three frozen joint task-split/model-seed
replicates: R0 (seed 17), R1 (seed 42), and R2 (seed 101). Each replicate
contains independent Pool A and Pool B models at `k = 1, 2, 4, 8, 16`, for 30
models and 960 validation model-task cells in total. Pool A and Pool B share no
task names within any replicate.

| k | Opposite-pool exact accuracy | Exact minus majority | Final-layer A-vs-B CKA |
|---:|---:|---:|---:|
| 1 | 12.25% +/- 2.66% | -20.59 +/- 2.66 pp | 0.172313 +/- 0.073809 |
| 2 | 11.49% +/- 2.22% | -21.34 +/- 2.22 pp | 0.191957 +/- 0.119854 |
| 4 | 11.96% +/- 1.90% | -20.87 +/- 1.90 pp | 0.212761 +/- 0.092239 |
| 8 | 13.64% +/- 3.73% | -19.19 +/- 3.73 pp | 0.619041 +/- 0.216940 |
| 16 | 16.72% +/- 2.50% | -16.11 +/- 2.50 pp | 0.496115 +/- 0.150254 |

Final-layer CKA has a positive but non-monotonic mean association with `k`
(Spearman rho 0.90). R0 and R1 peak at `k=8`; R2 increases monotonically and
peaks at `k=16`. Exact unseen-property execution improves most at `k=16`, but
it remains below the majority-answer baseline at every `k`. The study therefore
supports noisy representational convergence, not reliable hard zero-shot
generalization.

- [Aggregate report and provenance](replicates/README.md)
- [Per-replicate behavioral values](replicates/behavior_replicates.csv)
- [Per-replicate CKA values](replicates/cka_replicates.csv)
- [Frozen three-replicate protocol](../../PROPERTY32_REPLICATES.md)

The values above are mean plus/minus sample standard deviation over three
replicate-level measurements. Model seed and task split vary together, so the
error bars capture their combined variability rather than separating the two
sources. Test data were not used.

## Question

The earlier nested-task CKA experiment increased `k` by retaining every task
from the smaller model, so task overlap could itself make representations more
similar. This exploratory follow-up asks whether task-free representations
become more similar when independently trained models learn more properties
but share no task names.

Thirty-two scalar permutation properties are partitioned into Pool A and Pool
B. Ten independent Transformers are trained: one model for each pool at
`k = 1, 2, 4, 8, 16`. The equal-`k` A-vs-B comparison has zero task overlap.

## Original R0 pilot result

The behavioral and representation-level evidence is mixed rather than a
monotonic generalization trend.

- Opposite-pool exact accuracy averaged across A and B rises from 9.18% at
  `k=1` to 17.75% at `k=16`, but it remains below the task-specific majority
  baseline at every `k`.
- Final-layer A-vs-B linear CKA rises from 0.2238 at `k=1` to 0.4328 at
  `k=16`, with Spearman rho 0.80 against `k` rank.
- The CKA sequence is not monotonic. It peaks at 0.8033 for `k=8`, then falls
  to 0.4328 for `k=16`.
- A random-initialization cross-seed control has final-layer CKA 0.8851. High
  CKA therefore cannot by itself be interpreted as common mathematical
  knowledge; architecture and input geometry also produce high similarity.

This one-seed pilot provides a positive descriptive signal, but it does not
show stable representational convergence as the number of properties grows.

## Behavioral generalization

These are task-macro averages across the two directions: an A-trained model is
evaluated on all 16 B tasks, and a B-trained model is evaluated on all 16 A
tasks. None of those evaluation tasks occurred in that model's training set.

| k | Loss | Token accuracy | Exact accuracy | Majority baseline | Exact minus baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.7641 | 53.44% | 9.18% | 32.83% | -23.65 pp |
| 2 | 4.1919 | 54.76% | 9.59% | 32.83% | -23.24 pp |
| 4 | 3.3470 | 55.71% | 11.99% | 32.83% | -20.84 pp |
| 8 | 3.2959 | 54.42% | 9.36% | 32.83% | -23.48 pp |
| 16 | 2.2383 | 58.81% | 17.75% | 32.83% | -15.08 pp |

Loss improves overall, but exact accuracy is non-monotonic and never reaches a
constant-answer baseline. This is evidence against reliable hard zero-shot
execution of unseen properties. The model-task values and separate A/B macro
tables are in the [behavioral report](behavior/README.md) and
[raw CSV](behavior/MODEL_TASK_ACCURACIES.csv).

`token_accuracy` is teacher-forced over the scalar answer token and EOS. It is
partly inflated by EOS, so exact sequence accuracy is the primary behavioral
metric. The majority baseline predicts each property's most frequent answer
on the same 160 validation examples. Across individual properties, this
baseline ranges from 7.5% to 83.75%, making baseline control essential.

## Representation similarity

Linear CKA compares hidden states at `<ONE_END>` for the same 4,096
deterministic validation prefixes. The landmark is before the task token, so
the probe input contains a permutation but no requested operation.

| k | Final-layer A-vs-B linear CKA |
|---:|---:|
| 1 | 0.223846 |
| 2 | 0.330352 |
| 4 | 0.319226 |
| 8 | 0.803336 |
| 16 | 0.432796 |

- Spearman rho: 0.8000
- Pearson `r` against `log2(k)`: 0.6241
- `k=16` minus `k=1`: +0.20895
- Monotonic non-decreasing: false

The complete 84-row layerwise table, random control, within-pool overlap
controls, exact probe IDs, and hashes are in the
[CKA report](cka/README.md) and
[pairwise CSV](cka/pairwise_layer_cka.csv).

## Data and model

- 16,000,000 generated examples, 500,000 per property
- permutation length 2 through 30
- 15.68M train / 160k validation / 160k untouched test split
- all 16M answers and Passage encodings fully verified
- four-layer pre-LN causal Transformer
- `d_model=256`, eight heads, FFN width 1,024, context 128
- 3,240,448 trainable parameters, tied embeddings, dropout 0.1
- 20,000 AdamW updates, effective batch 64, bf16
- one model seed: 17

The full property definitions and frozen pool order are in
[PROPERTY32_PROTOCOL.md](../../PROPERTY32_PROTOCOL.md).

## Limitations

1. One seed provides no error bars. At least two additional seeds per cell are
   required before treating the trend as a population claim.
2. The fixed 20,000-update budget reduces per-property exposure from about
   1.28M examples at `k=1` to about 80k at `k=16`. Task diversity and exposure
   are therefore confounded.
3. The pools share no task names, but several properties are natural duals,
   such as descents/recoils and LIS/LDS. The `k=8` spike may reflect these
   symmetries rather than generic diversity.
4. The task tokens for opposite-pool properties are not semantically grounded
   during training. The weak hard-zero-shot result does not rule out
   information recoverable by linear probes or low-shot fine-tuning.
5. All reported model metrics and CKA probes use validation data. The test
   split remains unread and should be used only after a confirmatory protocol
   is frozen.

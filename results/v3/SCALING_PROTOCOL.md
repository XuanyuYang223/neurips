# Data and model scaling protocol

This protocol extends the completed v3 nested experiment into a two-by-two
factorial study of training-data scale and model capacity. The existing
1x-data/1x-model nested matrix is the baseline, so three additional matrices
are required.

| Condition | Training-data scale | Training steps | Transformer layers | MLP blocks |
|---|---:|---:|---:|---:|
| Existing baseline | 1x | 20,000 | 4 | 1 |
| `data10x_model1x` | 10x | 200,000 | 4 | 1 |
| `data1x_model2x` | 1x | 20,000 | 8 | 2 |
| `data10x_model2x` | 10x | 200,000 | 8 | 2 |

## Research question

The experiment tests whether the weak exact generalization result is primarily
limited by training data, model capacity, or an interaction between the two.
The four cells permit the following comparisons:

- the data effect at 1x capacity;
- the capacity effect at 1x data;
- the data effect at 2x capacity;
- the capacity effect at 10x data;
- the data-by-capacity interaction.

## Meaning of 10x data

The original corpus contains 9.8 million training records, but each base model
processed approximately 1.28 million sample presentations during 20,000
updates. Increasing only the number of records on disk while holding the update
count fixed would not provide ten times as much training information.

The 10x conditions therefore use both:

- a new deterministic 100-million-record v3 corpus, whose first 98 million
  records form the training split; and
- 200,000 optimizer updates, yielding approximately 12.8 million sample
  presentations per model.

The new corpus uses seed `20260831`. The original validation shard 098 and test
shard 099 remain frozen for all four factorial cells. They are not replaced by
the validation and test portions of the new corpus.

The 100M parent manifest SHA-256 is
`6bafe42be4adc2fd956275af171d9efece40357c6de9cc5791b5514bda34591f`.
It contains 1,000 shards and 5,000,000 records per task. The first 980 shards
provide exactly 98,000,000 training records; the remaining new-corpus shards
are intentionally unused so evaluation remains paired with the baseline.

## Meaning of 2x model

Capacity is increased by doubling depth while preserving the established
vocabulary, context length, width, attention-head count, FFN ratio, dropout,
and tied embeddings:

- the Transformer increases from four to eight pre-LN causal decoder layers;
- the causal MLP increases from one to two token-mixing blocks.

The resulting registered parameter counts are 3,463,424 to 6,622,464 for the
Transformer (1.912x) and 2,930,176 to 5,555,968 for the MLP (1.896x). "2x"
denotes the preregistered depth intervention; shared embedding and output
parameters do not double.

## Controlled variables

All four cells use the same nested task order, fixed holdouts, architectures,
seeds, effective batch size of 64, optimizer, learning-rate endpoints, sequence
length, and evaluation examples. The 10x schedules scale warmup, checkpoint,
and validation intervals by ten so their relative positions are unchanged.

The three fixed structured holdouts are `to_reduced_word`, `compose`, and
`to_lehmer`. Their task-macro exact sequence accuracy is the primary outcome.
The Boolean `parity` holdout is reported separately. Loss and teacher-forced
token accuracy are secondary diagnostics.

After base-model evaluation, the same frozen 20-example support sets may be
used to repeat the few-shot comparison. Linear probing remains a separate
experiment and is not part of this scaling protocol.

## Frozen configurations

- [`permutation_scaling_data10x_model1x.toml`](../../configs/permutation_scaling_data10x_model1x.toml)
- [`permutation_scaling_data1x_model2x.toml`](../../configs/permutation_scaling_data1x_model2x.toml)
- [`permutation_scaling_data10x_model2x.toml`](../../configs/permutation_scaling_data10x_model2x.toml)

Each configuration defines the same 30-run nested matrix: two architectures,
five task counts, and three seeds. Together the three new cells contain 90 new
base-model runs.

After a condition passes strict audit, evaluate its 30 nested models on the
frozen test split with:

```bash
permutation-evaluate \
  --config configs/permutation_scaling_data10x_model1x.toml \
  --matrix nested \
  --output-dir results/v3/scaling/data10x-model1x/evaluation
```

Use the corresponding config and output directory for the other two cells.

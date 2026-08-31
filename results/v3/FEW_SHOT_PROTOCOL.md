# Henry-style 20-shot adaptation protocol

This document freezes the follow-up motivated by Henry Kvinge's proposed
fine-tuning notion of generalization. It is separate from the completed v3
zero-shot result and does not change any base-model checkpoint or authenticated
v3 configuration.

**Completion status:** all 144 runs completed, passed strict audit, and were
evaluated once on their 5,000 target-task examples in test shard 099. Results
are reported in [fewshot/README.md](fewshot/README.md).

## Research question

The zero-shot experiment gives each held-out operation an opaque task token
whose meaning was never grounded during base training. Henry proposed a softer
and more informative test: give every pretrained model a very small labeled
support set from an unseen task, fine-tune at low learning rate, and compare it
with a randomly initialized model trained on the same support data.

The primary question is whether increasing base-training task diversity makes
the model easier to adapt with only 20 examples.

## Frozen matrix

The experiment uses the 30 audited v3 nested base models and the four fixed
training holdouts:

- `to_reduced_word`;
- `compose`;
- `parity`;
- `to_lehmer`.

Each base model is adapted separately to each task:

```text
30 base models x 4 holdout tasks = 120 pretrained adaptations
2 architectures x 4 tasks x 3 seeds = 24 random-init controls
total = 144 runs
```

For each task and seed, 20 support examples are selected uniformly without
replacement from train shards 000-097. The same seed-specific support set is
paired across all task counts and both architectures. No validation or test
example can enter a support set. The selected records are frozen in
[`manifests/permutation-v3-fewshot-support.json`](../../manifests/permutation-v3-fewshot-support.json).

## Optimization

All model parameters are updated. Every run uses 200 optimizer steps,
micro-batch 4, no gradient accumulation, AdamW, weight decay 0.01, 20 warmup
steps, cosine decay to 10% of the initial learning rate, gradient clipping at
1.0, and bfloat16 AMP.

- Pretrained models use Henry's low fine-tuning learning rate: `1e-5`.
- Random-init controls use the already established v3 from-scratch learning
  rate: `3e-4`. This gives the scratch baseline a reasonable optimizer rather
  than handicapping it with a fine-tuning rate.

Each adaptation sees 800 support-example presentations, exactly 40 passes over
its 20 examples. Training compute, examples, architecture, task, and support
set are matched. The progressive pretrained-model comparison changes only the
base training task count. The random control additionally uses its
preregistered from-scratch learning rate, which is reported explicitly rather
than being treated as an identical-optimizer comparison.

## Evaluation and aggregation

Every completed adaptation is first evaluated on all 5,000 examples for its
target task in validation shard 098. No early stopping or hyperparameter
selection is permitted. After all 144 checkpoints pass strict audit and the
implementation is committed, each model receives one evaluation on the 5,000
target-task examples in test shard 099.

The primary metric is exact canonical sequence accuracy. Loss and
teacher-forced token accuracy are secondary diagnostics. For each
architecture and base task count, the analysis:

1. macro-averages the four holdout tasks within each seed;
2. reports mean and sample standard deviation across three seeds;
3. reports improvement over the same base model's frozen zero-shot test result;
4. reports paired improvement over the random-init control using the same
   task, seed, support set, steps, and architecture.

This design measures few-shot adaptation, not zero-shot inference. Seen base
training tasks are never included in the adaptation average.

## Reproduction commands

```bash
# Build or verify the 12 paired support sets.
permutation-fewshot support \
  --config configs/henry_permutation_fewshot.toml

# Confirm the 120 + 24 matrix.
permutation-fewshot plan \
  --config configs/henry_permutation_fewshot.toml

# Train and validate all adaptations, then audit them.
permutation-fewshot run \
  --config configs/henry_permutation_fewshot.toml
permutation-fewshot audit \
  --config configs/henry_permutation_fewshot.toml

# Read test shard 099 only after the code/config/support commit is frozen.
permutation-fewshot test \
  --config configs/henry_permutation_fewshot.toml
permutation-fewshot-results \
  --config configs/henry_permutation_fewshot.toml
```

Linear probing is Henry's other proposed diagnostic. It remains a separately
preregistered analysis because it trains a probe on hidden states rather than
fine-tuning the base model and therefore requires a different data split and
statistical interpretation.

Frozen artifact identities:

- few-shot configuration SHA-256:
  `8151ca27cabc61db753714bc3003db88815553a3ad1220e6f54268bc72714ad0`;
- support artifact SHA-256:
  `a9463d6402dde0425048ff156f94a32e6d2115cc4cbed3788398333ae3aedc0a`;
- base v3 configuration SHA-256:
  `dd75f31277e42f554ed681beda44bb53f2d4f65089fd9583540b9e645c4f1b40`.

# Property32 Twenty-Shot Fine-Tuning Protocol

## Question

Does increasing the number of properties used during base-model training make
a Transformer easier to adapt to genuinely unseen permutation properties from
the opposite task pool?

This is Henry Kvinge's fine-tuning notion of generalization, applied to the
same 30 Transformers used by the zero-overlap behavioral, CKA, and linear-probe
analyses. The machine-readable protocol is frozen in
[`configs/property32_fewshot.toml`](configs/property32_fewshot.toml).

## Balanced target selection

Each of the three replicates partitions 32 properties into disjoint Pool A and
Pool B. A model trained on Pool A is adapted only to targets in Pool B, and
conversely. The first balanced four-property block of the opposite pool is
used at every `k = 1, 2, 4, 8, 16`. Each target block contains exactly one
local, positional, cycle, and global/run property. The selection rule was
fixed without consulting few-shot outcomes.

This produces 30 base models times four targets, or 120 warm-start
adaptations. For every replicate, pool direction, and target, a
seed-matched randomly initialized Transformer is trained on the identical
support set, adding 24 controls. The complete matrix has 144 runs.

## Adaptation

- 20 deterministic support examples per `(replicate, pool direction, target)`
  are selected only from train shards 000--195.
- Support examples are identical across `k` within a replicate and pool
  direction.
- Every parameter is updated for 200 steps with microbatch size four.
- Warm-start learning rate is `1e-5`; random initialization uses `3e-4`.
- AdamW, 20 warmup steps, cosine decay to 10% of the initial learning rate,
  gradient clipping at 1.0, weight decay 0.01, and bf16 match the earlier
  Henry-style protocol.
- Validation uses all 5,000 examples for the assigned property and is a
  diagnostic only; it does not select a checkpoint or hyperparameter.

## Final evaluation and statistical units

The earlier linear probe evaluated 4,096 examples selected exclusively from
source shard 198. To avoid reusing those model-evaluation examples, this study
uses only source shard 199 for its final endpoint. That shard supplies 2,500
examples per property and had not previously been used for model evaluation.

For every warm-start run, the same shard is also evaluated before adaptation
to obtain its paired zero-shot baseline. The primary outcome is the change in
complete-answer sequence accuracy after 20-shot adaptation. A second paired
contrast subtracts the matching random-initialization control.

Metrics are first macro-averaged across four target properties within each
model, then across Pool A and Pool B directions within a replicate. Mean and
sample standard deviation are finally reported across the three joint
task-split/model-seed replicates. Property rows and pool directions are not
treated as independent replicates.

This protocol tests few-shot adaptability. It is distinct from hard zero-shot
execution and from linear decodability of a frozen representation.

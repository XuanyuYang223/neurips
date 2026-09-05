# Permutation multitask generalization

This repository contains four permutation-language-model studies for
the NeurIPS workshop project:

- **v2 baseline:** 30 models trained on nested subsets of 20 tasks;
- **v3 revision:** 48 models trained after incorporating Henry Kvinge's
  feedback, including both a 30-model nested study and an 18-model
  encoding/statistics/algebra category comparison;
- **32-property zero-overlap study:** 30 independently trained Transformers
  across three frozen task-split/model-seed replicates and two disjoint task
  pools at `k = 1, 2, 4, 8, 16`, designed to remove the direct task-overlap
  confound from the CKA trend comparison;
- **combinatorial task-geometry study:** 48 single-task specialists plus 60
  fixed-four-task models, designed to separate mathematical correspondence
  from task count and test whether CKA recovers eight known relations.

All public documentation and result tables are in English. The generated
datasets and formal model checkpoints are published as versioned GitHub Release
assets; see [Datasets and model checkpoints](ARTIFACTS.md) for download,
checksum-verification, and extraction instructions. The repository also
contains their manifests, verification reports, complete metrics, and
reproducible aggregation code.

## Results

The shareable result packages are separated by protocol version:

- [v2 baseline results](results/v2/README.md), including all 600
  model-by-task validation rows;
- [v3 revised results](results/v3/README.md), including all 960 validation and
  960 independent-test model-by-task rows;
- [32-property zero-overlap results](results/property32-zero-overlap/README.md),
  including all 960 validation model-by-task rows, replicate-level summaries,
  error bars, and layerwise CKA;
- [three-replicate protocol](PROPERTY32_REPLICATES.md), with the frozen task
  splits, model seeds, and aggregation rules;
- [fixed-seed task-subset replicate protocol](PROPERTY32_SUBSET_REPLICATES.md),
  which isolates task-partition sensitivity with R0/R3/R4 at model seed 17;
- [linear-probing protocol](PROPERTY32_LINEAR_PROBING.md), which tests whether
  unseen property values become more linearly decodable as `k` increases;
- [Property32 twenty-shot protocol](PROPERTY32_FEWSHOT.md), which applies
  Henry's fine-tuning test to balanced opposite-pool targets;
- [Property32 twenty-shot results](results/property32-zero-overlap/fewshot/README.md),
  including all 144 model-task endpoints and paired controls;
- [Property32 matched-learning-rate sensitivity protocol](PROPERTY32_FEWSHOT_LR_SENSITIVITY.md),
  a validation-only completion of the initialization-by-learning-rate factorial;
- [Property32 matched-learning-rate sensitivity results](results/property32-zero-overlap/fewshot/lr-sensitivity/README.md),
  including all 288 endpoints and paired learning-rate interactions;
- [paired v3 5/20/100-shot curve](results/v3/fewshot/shot-curve/README.md),
  which varies the nested support size while retaining matched random controls;
- [v3 length-extrapolation results](results/v3/size-extrapolation/README.md),
  which evaluate the 30 nested models on permutations of length 31–40;
- [deadline-scoped k=16 scaling protocol](SCALING_K16.md), a three-seed 2x2
  data-by-depth factorial using both architectures;
- [completed k=16 scaling results](results/v3/scaling/k16/README.md), including
  all 24 model endpoints, paired factorial contrasts, error bars, and the
  frozen-test evaluation provenance;
- [relation-controlled CKA protocol](PROPERTY32_RELATION_CONTROLLED.md), which
  removes co-selected natural duals and crosses three low-correlation task
  selections with three model seeds;
- [property-pair CKA protocol](PROPERTY_PAIR_CKA.md), which compares known
  combinatorial duals with preregistered no-obvious-duality controls;
- [combinatorial task-geometry protocol](PROPERTY_TASK_GEOMETRY.md), the
  confirmatory single-task plus fixed-four-task relatedness study;
- [combinatorial task-geometry results](results/property-task-geometry/cka/README.md),
  with every specialist, bundle, symmetry-control, and random-initialization
  CKA comparison;
- [32-property protocol](PROPERTY32_PROTOCOL.md), with its frozen task pools,
  data specification, model design, and CKA analysis plan;
- [result-file index](results/README.md), which explains every CSV.
- [cross-domain paper synthesis](PAPER_STUDY_SYNTHESIS.md), which aligns the
  integer and permutation evidence without pooling incompatible metrics.
- [four-page paper figure set](paper/FIGURES.md), with two main-text composite
  figures, three supplementary diagnostics, captions, and LaTeX snippets.
- [final paper tables](paper/TABLES.md), with the selected main-text trend
  table, supplementary extension summary, and copy-ready LaTeX.
- [v3 category-model linear probes](results/v3/linear-probing/category/README.md),
  which compare a common 32-property probe battery across E4/S4/A4 training.
- [four-representation transfer protocol](REPRESENTATION_TRANSFER_PROTOCOL.md),
  which jointly trains the one-line row and descents column before evaluating
  all 32 representation-task combinations.
- [four-representation transfer results](results/representation-transfer/README.md),
  with all 96 unaveraged model-cell measurements and the complete 4 x 8
  transfer matrix.

The primary v3 generalization summaries exclude every task used to train the
model being evaluated. In other words, trained-task performance is retained
for diagnostics but is not included in the reported generalization average.

The v3 result is negative but useful: average loss and token accuracy often
improve on unseen tasks as the training set becomes more diverse, but exact
complete-answer accuracy remains very low and non-monotonic. The experiment
therefore does not demonstrate reliable zero-shot acquisition of an unseen
permutation operation. See the [v3 results](results/v3/README.md) for the full
loss and accuracy changes.

Henry's 20-shot follow-up is also complete. Transformer adaptation improves
with broader base training on the four-task macro, but the increase is almost
entirely due to Boolean `parity`; exact accuracy on the three structured
holdouts remains at or below 0.113%. See the
[few-shot report](results/v3/fewshot/README.md).

The [CKA representation analysis](results/v3/cka/README.md) is also complete.
Using the same 4,096 task-free validation prefixes for every model, it finds
no consistent monotonic increase in cross-seed representation similarity as
the nested task count grows. The Transformer becomes progressively closer to
its same-seed `k=16` reference, but that comparison is confounded by increasing
task overlap; the MLP does not show the same pattern.
The [disjoint-category follow-up](results/v3/cka/category/README.md) fixes
`k=4` and finds substantially higher CKA within the same training family than
between zero-overlap task families for both architectures.

The [four-representation transfer experiment](results/representation-transfer/README.md)
trains three joint Transformers on the one-line row and descents column of a
`4 representations x 8 tasks` grid. Exact accuracy is `54.31 +/- 0.96%` on
the 11 trained cells and `30.61 +/- 2.64%` on the 21 held-out cells. After
subtracting each cell's constant-answer majority baseline, held-out accuracy
remains `+11.46 +/- 2.64` percentage points. This is evidence of partial
cross-representation/task transfer, not uniformly successful transfer.

The [32-property zero-overlap study](results/property32-zero-overlap/README.md)
trains Pool A and Pool B independently at five values of `k` under three
frozen task-split/model-seed replicates. Mean opposite-pool exact accuracy is
12.25%, 11.49%, 11.96%, 13.64%, and 16.72%; it remains 16.11--21.34 percentage
points below a task-specific majority baseline. Mean final-layer A-vs-B CKA is
0.1723, 0.1920, 0.2128, 0.6190, and 0.4961. Its association with `k` is
positive (Spearman rho 0.90) but not monotonic. Two of three replicates peak at
`k=8`; the third increases monotonically and peaks at `k=16`.

The [fixed-seed task-subset extension](results/property32-zero-overlap/subset-replicates/README.md)
holds Transformer initialization at seed 17 while changing the balanced A/B
partition across R0, R3, and R4. Mean final-layer CKA is 0.101, 0.175, 0.302,
0.644, and 0.600 for `k = 1, 2, 4, 8, 16` (Spearman rho 0.90). R3 and R4
increase monotonically and peak at `k=16`, whereas R0 peaks at `k=8`.
Task-subset sample SD remains 0.11--0.18, so the positive association is more
replicable than strict monotonicity. Opposite-pool exact accuracy remains
below the majority baseline at every `k`.

Henry's [linear-probing follow-up](results/property32-zero-overlap/linear-probing/README.md)
finds a clearer but still non-monotonic internal signal. Final-layer
length-conditioned R2 on the 16 opposite-pool properties is 0.198, 0.245,
0.271, 0.307, and 0.297 for `k = 1, 2, 4, 8, 16`; a random Transformer reaches
0.215. All three replicates improve from `k=1` to `k=8`, but two decline at
`k=16`. The result supports progressive linear decodability through `k=8`,
not reliable hard zero-shot execution or a monotonic scaling law.

Henry's [Property32 twenty-shot follow-up](results/property32-zero-overlap/fewshot/README.md)
finds a stronger progressive adaptation signal. Exact accuracy after 20-shot
fine-tuning is 16.63%, 20.79%, 25.72%, 31.77%, and 33.59% for
`k = 1, 2, 4, 8, 16`, while paired gains over zero-shot increase from +3.48
to +17.80 percentage points. However, the support-matched random-init control
reaches 34.28%, and the warm-start models remain below it at every `k`.
Broader base training therefore predicts easier low-shot adaptation, but the
experiment does not establish a net advantage over training from scratch.

The validation-only [matched-learning-rate sensitivity](results/property32-zero-overlap/fewshot/lr-sensitivity/README.md)
shows that this comparison was materially confounded by optimization. At
`1e-5`, pretrained-minus-random exact accuracy changes from -5.30 percentage
points at `k=1` to +11.71 points at `k=16`. At `3e-4`, the corresponding
contrast changes from -0.36 to only +2.96 points and is non-monotonic between
the endpoints. Progressive low-rate adaptation therefore survives a matched
comparison, but its magnitude is learning-rate dependent; this post-hoc result
uses validation only and is not a second confirmatory test result.

The completed [k=16 data-by-depth factorial](results/v3/scaling/k16/README.md)
finds that additional scale does not rescue hard zero-shot execution. Across
all 24 architecture-condition-seed endpoints, exact accuracy is 0% on each of
the three structured training holdouts: reduced-word translation,
composition, and Lehmer-code translation. Tenfold training exposure, doubled
depth, and their combination therefore all have a 0.000 +/- 0.000 percentage
point effect on the primary exact-accuracy macro. Secondary token accuracy and
loss move in different directions across architectures, so they do not
support a general scaling improvement. Boolean parity remains low and
variable and is reported separately from the structured macro.

The [combinatorial task-geometry study](results/property-task-geometry/cka/README.md)
provides a more controlled answer. Across single-task specialists, eight
preregistered directly related task pairs have higher final-layer CKA than the
112 other cross-task pairs (0.1375 versus 0.0903; task-label permutation
`p=0.015`). The symmetry control is stronger: using the mathematically correct
inverse or complement raises CKA over both the identity and wrong-transform
controls in all 24 pair-seed units and, after aggregating seeds, in all eight
mathematical relations (relation-level two-sided sign test `p=0.0078` for both
contrasts). However, the fixed-four-task experiment
does not show a monotonic dose response as the number of direct
correspondences increases (`r=0,1,2,4`: 0.2935, 0.2752, 0.2662, 0.3648;
paired `r=4-r=0` sign-test `p=0.774`). The evidence therefore supports
transformation-specific alignment, not a general claim that adding related
tasks always makes learned representations more similar.

## v3 revision

Henry suggested removing the unusually slow `power`, `conjugate`, and
`commutator` tasks and comparing models trained by task family. We replaced
them with three linear-time statistics while retaining 20 balanced tasks:

| Removed v2 task | Added v3 task | v3 records |
|---|---|---:|
| `power` | `peaks` | 500,000 |
| `conjugate` | `exceedances` | 500,000 |
| `commutator` | `recoils` | 500,000 |

The v3 corpus contains 10,000,000 records: 500,000 per task, permutation size
`2 <= n <= 30`, and a 9.8M/100k/100k train/validation/test split. Every record
passed mathematical-answer recomputation and canonical-encoding verification.
The manifest snapshot and verification report are available at
[manifests/permutation-10m-v3.json](manifests/permutation-10m-v3.json) and
[manifests/permutation-10m-v3-verification.json](manifests/permutation-10m-v3-verification.json).

## Architecture and encoding

Both model families use a 163-token vocabulary, a 1,024-token context,
`d_model=256`, tied input/output embeddings, and bfloat16 training.

| Architecture | Configuration | Parameters |
|---|---|---:|
| Transformer | Standard pre-LN causal decoder, 4 layers, 8 heads, FFN width 1,024 | 3,463,424 |
| MLP | Strictly causal token-mixing block plus a position-wise channel MLP | 2,930,176 |

Each example is encoded as one causal Passage Math sequence:

```text
<BOS> <SIZE> n <ONE_START> pi(1) , ... , pi(n) <ONE_END>
<TASK_TOKEN> [typed operand] = <canonical answer> <EOS>
```

Values `0` through `99` are atomic two-character tokens such as `00`; they do
not use `<NUM_START>`. Values at least 100 use base-100 digits between
`<NUM_START>` and `<NUM_END>`. Exact task definitions and output conventions
are in [PROTOCOL.md](PROTOCOL.md).

## Repository map

- [results/](results/README.md): versioned, shareable experiment results;
- [20-shot follow-up](results/v3/FEW_SHOT_PROTOCOL.md): Henry-style low-LR
  adaptation against a random-initialization control;
- [20-shot results](results/v3/fewshot/README.md): 144 audited adaptations and
  paired zero-shot/random-init comparisons;
- [CKA results](results/v3/cka/README.md): layerwise representation similarity
  across nested task counts, seeds, architectures, and random-init controls;
- [32-property zero-overlap protocol](PROPERTY32_PROTOCOL.md): 32 scalar
  properties and two disjoint balanced pools;
- [three-replicate protocol](PROPERTY32_REPLICATES.md): the 30-model
  confirmatory extension and frozen aggregation;
- [fixed-seed task-subset extension](PROPERTY32_SUBSET_REPLICATES.md): two
  additional balanced partitions for separating task-selection variability;
- [fixed-seed task-subset results](results/property32-zero-overlap/subset-replicates/README.md):
  R0/R3/R4 behavioral and CKA curves with initialization held fixed;
- [linear-probing protocol](PROPERTY32_LINEAR_PROBING.md): task-free layerwise
  probes for all 32 properties across the 30 zero-overlap Transformers;
- [Property32 twenty-shot protocol](PROPERTY32_FEWSHOT.md): 120 warm-start
  adaptations and 24 matched random-initialization controls;
- [Property32 twenty-shot results](results/property32-zero-overlap/fewshot/README.md):
  complete test metrics, three-replicate error bars, and family breakdowns;
- [Property32 matched-learning-rate sensitivity](PROPERTY32_FEWSHOT_LR_SENSITIVITY.md):
  the frozen exploratory protocol for separating initialization from learning rate;
- [Property32 matched-learning-rate results](results/property32-zero-overlap/fewshot/lr-sensitivity/README.md):
  288 validation endpoints, matched contrasts, and interaction effects;
- [k=16 scaling results](results/v3/scaling/k16/README.md): 24 independently
  evaluated endpoints from the three-seed 2x2 exposure-by-depth factorial;
- [relation-controlled protocol](PROPERTY32_RELATION_CONTROLLED.md): the
  72-cell/60-model low-correlation 3x3 CKA follow-up;
- [property-pair CKA protocol](PROPERTY_PAIR_CKA.md): the controlled
  known-related versus no-obvious-duality representation analysis;
- [four-representation transfer results](results/representation-transfer/README.md):
  complete one-line/cycle/Lehmer/inversion-vector by eight-task test matrix;
- [combinatorial task-geometry protocol](PROPERTY_TASK_GEOMETRY.md): the
  108-model specialist and fixed-task-count composition study;
- [TRAINING_PROCESS.md](TRAINING_PROCESS.md): full data, architecture,
  training, recovery, audit, and evaluation record;
- [EXPERIMENTS.md](EXPERIMENTS.md): experimental designs and limitations;
- [PROTOCOL.md](PROTOCOL.md): mathematical and tokenization specification;
- [configs/](configs): frozen v2 and v3 experiment configurations;
- [src/neurips_permutations/](src/neurips_permutations): generator, verifier,
  models, trainer, evaluator, auditor, and result aggregator.

## Reproduce the software checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest

# Small deterministic smoke corpus.
permutation-generate --count 2000 --output-dir data/smoke --workers 1
permutation-verify data/smoke/manifest.json --full

# Rebuild the authenticated v3 result tables from completed local artifacts.
permutation-results \
  --config configs/henry_permutation_revised.toml \
  --output-dir results/v3 \
  --test-evaluation-dir results/v3/evaluation

# Recompute CKA from the 30 completed nested checkpoints and validation shard 098.
permutation-cka \
  --config configs/henry_permutation_revised.toml \
  --output-dir results/v3/cka \
  --probe-count 4096 \
  --device auto

# Run or resume all three frozen zero-overlap replicates, then aggregate them.
permutation-property-replicates --replicate all --run
permutation-property-replicate-results \
  --output-dir results/property32-zero-overlap/replicates \
  --probe-count 4096 \
  --device auto

# After freezing the protocol/code, fit validation probes and evaluate them
# once on the independently selected property test examples.
permutation-property-linear-probe \
  --config configs/property32_linear_probe.toml \
  --device auto

# Run Henry's balanced 20-shot follow-up on the Property32 base models.
permutation-property-fewshot support --config configs/property32_fewshot.toml
permutation-property-fewshot run --config configs/property32_fewshot.toml --device cuda
permutation-property-fewshot audit --config configs/property32_fewshot.toml
permutation-property-fewshot test --config configs/property32_fewshot.toml --device cuda
permutation-property-fewshot-results --config configs/property32_fewshot.toml

# Complete the validation-only initialization-by-learning-rate sensitivity.
permutation-property-fewshot-lr-sensitivity run --device cuda
permutation-property-fewshot-lr-sensitivity audit
permutation-property-fewshot-lr-sensitivity results

# Complete, audit, evaluate, and summarize the deadline-scoped k=16 factorial.
permutation-scaling-k16 run --config configs/permutation_scaling_k16.toml
permutation-scaling-k16 audit --config configs/permutation_scaling_k16.toml
permutation-scaling-k16 evaluate --config configs/permutation_scaling_k16.toml --device cuda
permutation-scaling-k16 results --config configs/permutation_scaling_k16.toml

# Regenerate the compact paper figure set from committed result CSVs.
pip install -e '.[figures]'
permutation-paper-figures --repository . --output-dir paper/figures
```

Production data shards and checkpoints are intentionally ignored by Git. The
checked-in CSVs are derived from authenticated completion markers and frozen
test evaluations: v3 shard 099 and the preregistered Property32 source shards
198 (linear probing) and 199 (twenty-shot adaptation).

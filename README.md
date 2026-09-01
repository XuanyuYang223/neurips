# Permutation multitask generalization

This repository contains two completed permutation-language-model studies for
the NeurIPS workshop project:

- **v2 baseline:** 30 models trained on nested subsets of 20 tasks;
- **v3 revision:** 48 models trained after incorporating Henry Kvinge's
  feedback, including both a 30-model nested study and an 18-model
  encoding/statistics/algebra category comparison.

All public documentation and result tables are in English. The large generated
datasets and model checkpoints remain local research artifacts; the repository
contains their manifests, verification reports, complete metrics, and
reproducible aggregation code.

## Results

The shareable result packages are separated by protocol version:

- [v2 baseline results](results/v2/README.md), including all 600
  model-by-task validation rows;
- [v3 revised results](results/v3/README.md), including all 960 validation and
  960 independent-test model-by-task rows;
- [result-file index](results/README.md), which explains every CSV.

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
```

Production data shards and checkpoints are intentionally ignored by Git. The
checked-in CSVs are derived from authenticated completion markers and the
single frozen evaluation of test shard 099.

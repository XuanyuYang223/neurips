# Permutation Multitask Experiments: Complete Data and Training Process

This document records the complete permutation workflow, from interpreting the requirements through data generation, encoding, model design, formal training, checkpoint recovery, verification, auditing, and publication of results. Sections 1–14 describe the completed v2 baseline; Section 0 records the completed v3 revision following Henry's feedback, including all 48 trained models and the independent test pass.

Last updated: 2026-08-31

- GitHub: <https://github.com/XuanyuYang223/neurips>
- Completed baseline: data protocol `permutation-20/v2`, experiment protocol `henry-permutation/v1`
- Baseline status: 30/30 base models completed; strict audit: 30 passed / 0 failed
- Revised main study: data protocol `permutation-20/v3`, experiment protocol `henry-permutation/v2-revised`
- Revised data status: 10,000,000 records, 100 shards, full verification passed
- Revised model status: **48/48 completed; strict audit 48 passed / 0 failed**
- Revised test status: one frozen shard099 pass, 4,800,000 model-examples
- Implementation commit used for formal v2 training: `6a40235`
- Post-training audit/results baseline commit: `32ff22a`

## 0. Revised Main Study After Henry's Feedback (v3)

Henry recommended keeping the architecture as standard as possible rather than reproducing the original PermuFormer exactly. He also recommended excluding the costly-to-learn `power`, `conjugate`, and `commutator` tasks from the paper's main experiments and comparing learned representations across task categories. Henry did not prescribe replacements; to retain 20 balanced tasks, this project selected `peaks`, `exceedances`, and `recoils`.

Accordingly, v3 retains the current standard small pre-LN decoder-only Transformer (4 layers, 8 heads, `d_model=256`) and the MLP control, without introducing new Transformer modifications. The task substitutions are:

| Removed from v2 | Added in v3 | Definition | New records |
|---|---|---|---:|
| `power` | `peaks` | Number of interior local peaks | 500,000 |
| `conjugate` | `exceedances` | Number of positions satisfying `pi(i) > i` | 500,000 |
| `commutator` | `recoils` | Number of descents of `pi^-1` | 500,000 |

All three new tasks are scalar properties for `n <= 30`; each answer uses a single `00`–`99` token and does not require `<NUM_START>`. The formal v3 dataset is complete: 20 tasks × 500,000 records = 10,000,000 records, written to the separate directory `data/permutation-10m-v3` without overwriting the v2 data.

| V3 data fact | Value |
|---|---:|
| Records | 10,000,000 |
| Shards | 100 |
| Compressed bytes | 1,139,175,228 |
| Generation wall time | 43.76 seconds |
| Full verification wall time | 34.59 seconds |
| Parent manifest SHA-256 | `b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f` |

V3 split:

| Split | Shards | Records | Manifest SHA-256 |
|---|---|---:|---|
| Train | `000-097` | 9,800,000 | `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa` |
| Validation | `098` | 100,000 | `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068` |
| Test | `099` | 100,000 | `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b` |

The revised nested matrix used `1/2/4/8/16 tasks × 2 architectures × 3 seeds = 30` completed models and retained the same four holdouts as v2: `to_reduced_word`, `compose`, `parity`, and `to_lehmer`. Keeping the same holdout identities enables direct v2–v3 comparison while preserving one algebra holdout. The revised outputs were written separately to `runs/henry-permutation-v3`.

To address Henry's question about category-level representational differences, this project operationalizes it as three task-count-matched groups:

| Condition | Four training tasks |
|---|---|
| Encoding E4 | `to_cycle`, `to_lehmer`, `to_inversion_vector`, `to_reduced_word` |
| Statistics S4 | `length`, `cycle_type`, `rsk_shape`, `pattern_avoidance` |
| Algebra A4 | `inverse`, `compose`, `right_multiply_simple`, `bruhat_leq` |

This avoids the task-count confound that would arise from directly comparing the full `4 vs 12 vs 4` category sets. The three groups used identical per-task data, optimizer-update budgets, architectures, and three seeds, for a total of `3 categories × 2 architectures × 3 seeds = 18` completed models. Layerwise CKA, frozen linear probes, and few-shot transfer remain planned at the `<ONE_END>` position for the same held-out one-line permutations.

The authoritative design is documented in [configs/henry_permutation_revised.toml](configs/henry_permutation_revised.toml) and [EXPERIMENTS.md](EXPERIMENTS.md). Both the 30-run nested matrix and the isolated 18-run E4/S4/A4 category matrix used schema-aware plan, dry-run, resume, and strict-audit support. E4 used micro-batches of 4 with 16-way gradient accumulation, while S4/A4 used 16 with 4-way accumulation, giving all category conditions 64 examples per optimizer update despite long reduced-word sequences. All 48 completion markers passed strict audit, and the frozen shard099 evaluation processed 100,000 examples per model. Complete results are in [results/v3/README.md](results/v3/README.md).

Henry's fine-tuning proposal was then implemented as a separately frozen 20-shot follow-up. The 30 nested base models were each adapted independently to four holdouts, producing 120 warm-start runs, and 24 architecture/task/seed-matched random-initialization controls were added. Every run used 20 train-split support examples, 200 optimizer steps, and a full 5,000-example target-task validation before strict audit. All 144 checkpoints passed; the frozen test pass then evaluated 720,000 model-examples. Transformer four-task exact accuracy increased with base-task count, but almost all complete-answer success came from Boolean `parity`; structured-task exact accuracy remained at or below 0.113%. Full results and provenance are in [results/v3/fewshot/README.md](results/v3/fewshot/README.md).

## 1. V2 Baseline Research Objective and Scope of Completion

Henry's central question is whether a model's internal representations and generalization ability improve systematically as the variety of tasks on which it is trained increases. The v2 permutation baseline uses nested task sets and compares the Transformer and MLP under an identical optimizer-step budget.

The v2 baseline completed the following:

1. Defined and implemented 20 permutation tasks.
2. Generated 10,000,000 Passage Math training sequences.
3. Recomputed the mathematical answer for every record and verified its encoding.
4. Trained `1/2/4/8/16 tasks × 2 architectures × 3 seeds = 30` base models.
5. Saved resumable checkpoints, final completion markers, and validation metrics.
6. Performed strict structural, hash, and numerical audits on all 30 checkpoints.
7. Summarized preliminary zero-shot generalization results.

The v2 baseline has not yet completed:

- A frozen evaluation on test shard 099, which has never been used for model evaluation.
- Few-shot fine-tuning on the holdout tasks.
- A comparison against a few-shot baseline trained from random initialization.
- Linear probing and representation geometry analyses.
- The full `4 representations × 8 tasks` input-combination experiment from the second proposal.

Thus, the v2 baseline completes the task-selection, data-generation, and base-model-training stages of Henry's proposal, together with a preliminary validation-set zero-shot result; it cannot yet be described as a complete generalization study. The v3 study separately completes data generation, 48 base models, independent behavioral test evaluation, and the 144-run 20-shot adaptation comparison, while representation analyses remain outstanding.

## 2. Requirement Interpretation and Frozen Decisions

### 2.1 `maximum entries = 30`

Every object is a permutation in the standard symmetric group `S_n`:

```text
pi is a permutation of {1, 2, ..., n}, with 2 <= n <= 30.
```

Therefore, the maximum permutation length is 30, and the maximum entry is naturally also 30. We did not interpret this as a partial permutation consisting of 30 distinct numbers selected from 1 through 100, because operations involving cycles, Bruhat order, and Coxeter generators require the standard `S_n` semantics.

### 2.2 `maximum number = 100`

In the v2 baseline, 100 was implemented as the user-specified base-100 number-tokenizer convention and the upper bound on the exponent in the power task, rather than as an upper bound on permutation entries:

- `00` through `99` are atomic number tokens.
- Values of 100 or greater use `<NUM_START> ... <NUM_END>`.
- The v2 power exponent satisfies `0 <= k <= 100`; v3 removes the power task, but the number encoding is unchanged.

### 2.3 Unit of “10M data”

10M means 10,000,000 final causal-LM sequences. Each sequence contains exactly one task and one answer.

It does not mean:

```text
10M base permutations × 20 labels = 200M model sequences
```

The final v2 dataset is exactly balanced across 20 tasks, with 500,000 records per task. V3 has also been completed under the same total-size and balance constraints, including 500,000 records for each of the three new properties. This choice avoids the hundreds of gigabytes of storage and extremely long training time associated with roughly 200M rich JSON records while preserving task balance.

### 2.4 Input representation

The primary input for all 30 current Henry base models always uses one-line notation. Cycle notation, Lehmer code, inversion vector, and reduced Coxeter word are translation targets.

The `4 input representations × 8 tasks = 32 combinations` proposed in the second attachment constitute a separate representation-transfer experiment; they were not mixed with the 20-task Henry nested matrix in this round.

## 3. The Twenty V2 Baseline Tasks

The following list describes the completed v2 training setup. V3 replaces `power`, `conjugate`, and `commutator` with `peaks`, `exceedances`, and `recoils`, respectively; see Section 0 and [PROTOCOL.md](PROTOCOL.md) for the complete v3 registry.

### 3.1 Encoding / translation (4)

1. `to_cycle`: canonical disjoint cycles, including singleton cycles; each cycle begins with its smallest element, and cycles are ordered by their smallest elements.
2. `to_lehmer`: `L_i = #{j > i : pi_j < pi_i}`.
3. `to_inversion_vector`: value-indexed, `I_v = #{u > v : position(u) < position(v)}`.
4. `to_reduced_word`: a deterministic reduced adjacent-generator word produced by stable bubble sort.

### 3.2 Statistics / properties (9)

5. `length`: Coxeter length / inversion count.
6. `descents`: number of descents.
7. `fixed_points`: number of fixed points.
8. `parity`: inversion count modulo 2, with `00=even` and `01=odd`.
9. `cycle_type`: cycle lengths, including 1-cycles, in descending order.
10. `rsk_shape`: row lengths of the RSK insertion tableau.
11. `lis_length`: longest strictly increasing subsequence length.
12. `lds_length`: longest strictly decreasing subsequence length.
13. `pattern_avoidance`: whether the permutation avoids a given classical pattern, with `01=avoids` and `00=contains`.

### 3.3 Algebraic operations / comparisons (7)

We use the composition convention:

```text
(a o b)(i) = a(b(i))
```

14. `inverse`: `pi^-1`.
15. `compose`: `pi o sigma`.
16. `power`: `pi^k`, with `0 <= k <= 100`.
17. `conjugate`: `g o pi o g^-1`.
18. `commutator`: `pi o sigma o pi^-1 o sigma^-1`.
19. `right_multiply_simple`: `pi s_i`, which swaps one-line positions `i` and `i+1`.
20. `bruhat_leq`: strong Bruhat comparison.

The authoritative mathematical definitions and conventions are given in [PROTOCOL.md](PROTOCOL.md) and [math_ops.py](src/neurips_permutations/math_ops.py).

## 4. Passage Math Encoding

### 4.1 Vocabulary

The formal v2 vocabulary size is 163:

- 100 number tokens: `00`–`99`.
- 36 original fixed tokens.
- 27 task, operand, and structured-answer tokens added for the 20-task dataset.

The input embedding and output LM head use the same vocabulary and are weight-tied.

### 4.2 Number encoding

```text
0     -> 00
7     -> 07
28    -> 28
99    -> 99
100   -> <NUM_START> 01 00 <NUM_END>
137   -> <NUM_START> 01 37 <NUM_END>
9999  -> <NUM_START> 99 99 <NUM_END>
10000 -> <NUM_START> 01 00 00 <NUM_END>
```

This gives every nonnegative integer a unique canonical encoding.

### 4.3 General sequence grammar

```text
<BOS> <SIZE> ENCODE(n)
<ONE_START> PRIMARY <ONE_END>
[TYPED OPERANDS]
<TASK> = TYPED_ANSWER
<EOS>
```

Each sequence contains exactly one task token and one answer. The training loss supervises only the answer tokens and `<EOS>`; labels for the prompt, primary permutation, operands, task token, and `=` are all set to the ignore index.

### 4.4 Representation example

Let `pi = [3,1,4,2]`.

One-line:

```text
<ONE_START> 03 , 01 , 04 , 02 <ONE_END>
```

Canonical cycle:

```text
<CYCLE_START> 01 , 03 , 04 , 02 <CYCLE_END>
```

Lehmer code `[2,0,1,0]`:

```text
<LEHMER_START> 02 , 00 , 01 , 00 <LEHMER_END>
```

Inversion vector `[1,2,0,0]`:

```text
<INVEC_START> 01 , 02 , 00 , 00 <INVEC_END>
```

Reduced word `[2,3,1]`:

```text
<REDUCED_WORD_START> 02 , 03 , 01 <REDUCED_WORD_END>
```

Complete translation-sequence example:

```text
<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END>
<TO_LEHMER> = <LEHMER_START> 02 , 00 , 01 , 00 <LEHMER_END> <EOS>
```

### 4.5 Typed operands

```text
Second permutation: <ARG_START> ... <ARG_END>
Pattern:            <PATTERN_START> ... <PATTERN_END>
Exponent:           <EXPONENT> ENCODE(k)
Simple index:       <SIMPLE_INDEX> ENCODE(i)
```

The encoding implementation is in [passage.py](src/neurips_permutations/passage.py).

### 4.6 JSONL record schema

Each line stores both structured fields and the final token sequence:

```json
{
  "schema_version": "permutation-20/v2",
  "id": 0,
  "task": "...",
  "n": 12,
  "inputs": {"primary": [1, 2, 3]},
  "answer": 0,
  "answer_kind": "scalar",
  "tokens": ["<BOS>", "...", "<EOS>"],
  "canonical_text": "<BOS> ... <EOS>"
}
```

Depending on the task, `answer` and `inputs` contain a scalar, Boolean, permutation, pattern, exponent, or nested lists.

## 5. Data Generation

### 5.1 Deterministic sampling

- Global seed: `20260830`.
- The RNG for each record is determined by the global seed and record ID.
- Tasks rotate exactly according to `record_id mod 20`.
- For most tasks, `n` is sampled from 2–30.
- Pattern avoidance uses `n >= 3`.
- Bruhat order uses `n >= 4`.
- Duplicates are permitted because this is synthetic sampling with replacement.

Positive and negative pattern-avoidance examples are exactly balanced. Pattern length is `n-1`: for label `00` (contains), the pattern is selected from the standardized deletion patterns obtained by deleting one entry from the primary permutation; for label `01` (avoids), the pattern is guaranteed not to belong to that set.

Positive and negative Bruhat examples are also exactly balanced and matched for permutation size and positive Coxeter-length gap. The gap is 1–4 (1–2 in `S_4`); positive examples are constructed along strong Bruhat covers, while negative examples are incomparable pairs with the same gap. Therefore, the label cannot be inferred from the inversion-length gap alone.

### 5.2 Streaming and sharding

The generator never loads all 10M records into memory:

- 100 shards.
- 100,000 records per shard.
- deterministic `jsonl.gz`.
- gzip level 6, with the header timestamp fixed at 0.
- Each shard is first written to a temporary file, flushed, passed through `fsync`, and then atomically renamed.
- The manifest stores record ranges, byte sizes, SHA-256 hashes, and task counts.
- On resume, a shard is reused only if its checksum, count, and configuration all match.

### 5.3 v1 Review and v2 Regeneration

Review of the initial dataset identified direct leakage in the Bruhat labels: every positive v1 example had an inversion-length gap of `+1`, while every negative example had a gap of `-1`, so the label could be read directly from the direction. The early verifier also checked only whether the stored answer could be rendered again; it did not recompute the mathematical ground truth from the inputs.

Two corrections were made before formal training:

1. Bruhat examples were changed to the matched-gap comparable/incomparable construction described above.
2. The full verifier now calls the authoritative `math_ops` implementation to recompute the answers for all 20 tasks from the inputs and then reconstructs the tokens from that ground truth; `math_ops` is itself cross-checked by exhaustive small-`n` tests and unit tests.

After these corrections, `permutation-20/v2` was regenerated. The old directory `data/permutation-10m` was not used for any formal training; all formal runs read only `data/permutation-10m-v2`.

### 5.4 Final data volume

| Split | Shards | Records | Per task | Compressed bytes |
|---|---:|---:|---:|---:|
| Train | 98 (000–097) | 9,800,000 | 490,000 | 1,263,940,793 |
| Validation | 1 (098) | 100,000 | 5,000 | 12,866,288 |
| Test | 1 (099) | 100,000 | 5,000 | 12,911,897 |
| Total | 100 | **10,000,000** | **500,000** | **1,289,718,978** |

The total compressed size is 1.290 GB. The gzip-decompressed JSONL occupies 8,734,219,058 bytes, or 8.734 GB, with an average of approximately 873 bytes per record.

Parent manifest SHA-256:

```text
a9cc873bc82777c50fc2cfced96f54d727e3c3964eff457bd1a03ffabb179e87
```

### 5.5 Data verification

The final full verification used 20 workers and performed the following checks:

- SHA-256 hash and byte size of all 100 shards.
- Continuity of IDs, shard indices, and record counts.
- Schema, task balance, and permutation validity.
- Recalculation from the inputs of all 20 task answers for all 10,000,000 records.
- Consistency among the stored answer, typed answer kind, tokens, and canonical text.
- The parent full verification checks all physical shards; parent metadata, ranges, and counts for split views are checked separately by the split verifier/tests and formal audit.

Result: 10,000,000 / 10,000,000 records passed. Generation took approximately 41.4 seconds, and the complete mathematical verification took approximately 32.0 seconds.

## 6. Model Architecture

Both models implement the same interface:

```text
forward(input_ids, attention_mask) -> logits[B, L, 163]
```

Shared components:

- learned token embedding.
- learned absolute position embedding.
- maximum context length 1024.
- `d_model = 256`.
- pre-LayerNorm, GELU, and residual connections.
- channel MLP hidden dimension 1024.
- dropout 0.1.
- final LayerNorm.
- bias-free tied LM head.
- strict prefix causality and padding masking.

| Architecture | Blocks | Attention | Token mixing | Parameters |
|---|---:|---|---|---:|
| Causal Transformer | 4 | 8 heads, head dim 32 | causal self-attention | 3,463,424 |
| Causal MLP | 1 | none | two masked `1024 × 1024` linear maps | 2,930,176 |

### 6.1 Transformer

Each block:

```text
x = x + CausalSelfAttention(LayerNorm(x))
x = x + ChannelMLP(LayerNorm(x))
```

Attention uses an explicit lower-triangular causal mask and a padding-key mask. The channel MLP is `256 -> 1024 -> 256`.

### 6.2 Causal MLP

The MLP has neither attention nor recurrence. Each block is:

```text
x = x + CausalTokenMixingMLP(LayerNorm(x))
x = x + ChannelMLP(LayerNorm(x))
```

The token-mixing MLP contains two learned `1024 × 1024` matrices; only their lower-triangular entries are used during the forward pass, with GELU between the two layers. The matrices are shared across channels, making the structure similar to a causal MLP-Mixer. Prefix-invariance tests verify that suffix tokens cannot alter any earlier representation.

Only one MLP block is used so that its registered parameter count under a 1024-token context is approximately matched to that of the 4-layer Transformer; this is not a claim that a 1-layer MLP is depth-equivalent to a 4-layer Transformer. The figure 2,930,176 is the nominal/registered count: 1,047,552 strict upper-triangular parameters in the two `1024 × 1024` matrices are masked by causality and are not used in the forward pass, so the registered count is not equal to the active degrees of freedom.

See [models.py](src/neurips_permutations/models.py) for the implementation.

## 7. Completed Henry v2 Nested-Task Experiment Matrix

### 7.1 Frozen task order

The tasks were frozen in the following order using seed `20260830`, rather than retaining the order in which they were originally proposed:

```text
 1  power
 2  lis_length
 3  fixed_points
 4  to_inversion_vector
 5  pattern_avoidance
 6  lds_length
 7  right_multiply_simple
 8  conjugate
 9  to_cycle
10  descents
11  commutator
12  length
13  rsk_shape
14  inverse
15  cycle_type
16  bruhat_leq
17  to_reduced_word       fixed holdout
18  compose               fixed holdout
19  parity                fixed holdout
20  to_lehmer             fixed holdout
```

### 7.2 Nested subsets

| Subset | Training tasks |
|---:|---|
| 1 | tasks 1 |
| 2 | tasks 1–2 |
| 4 | tasks 1–4 |
| 8 | tasks 1–8 |
| 16 | tasks 1–16 |

The final four tasks are held out from training sequences and gradient updates for all base models. However, the validation diagnostics evaluate them every 1,000 steps, so they do not constitute an unseen test set.

### 7.3 Run count

```text
5 subset sizes × 2 architectures × 3 seeds = 30 formal runs
```

Model seeds: `17`, `42`, and `314159`. The three seeds alter initialization and streaming shuffle, but the task order itself was frozen only once.

The complete configuration is in [configs/henry_permutation.toml](configs/henry_permutation.toml), whose SHA-256 is:

```text
c5d9a0ea7a601588d1e07a520721dfeb3b8f96830d03c8c9f8632c6d37f70dfa
```

## 8. Training Configuration

| Item | Value |
|---|---:|
| Optimizer | AdamW |
| Optimizer steps / run | 20,000 |
| Micro-batch max examples | 16 |
| Gradient accumulation | 4 |
| Nominal examples / optimizer step | 64 |
| Max padded tokens / micro-batch | 4,096 |
| Initial learning rate | 0.0003 |
| Warmup | 1,000 steps |
| Schedule | cosine decay |
| Minimum LR ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Precision | bf16 AMP |
| Shuffle buffer | 10,000 records |
| Checkpoint interval | 1,000 steps |
| Validation interval | 1,000 steps |
| Validation batches | 5 dynamic batches / task |
| DataLoader workers | 0 |

### 8.1 Streaming loader

The training loader streams gzip shards directly, filters records by the current run's tasks, and uses a bounded deterministic shuffle without loading the full dataset into memory. `TokenBudgetBatcher` constrains both example count and padded-token count, so long sequences such as reduced words automatically use a smaller micro-batch instead of being truncated into incorrect answers.

### 8.2 Answer-only objective

The model performs causal next-token prediction, but the loss covers only the answer and `<EOS>`:

```text
prompt labels -> -100 / ignored
answer labels -> supervised
EOS label      -> supervised
```

The loss is first averaged over the supervised tokens within each example and then averaged across examples. Consequently, a reduced word that may contain hundreds of tokens does not receive tens of times more weight than a scalar task merely because its answer is longer.

### 8.3 Fixed update budget

All models receive the same 20,000 optimizer updates, rather than assigning the same number of steps to each task. This controls total computation, but the number of examples allocated to each task decreases as the number of tasks increases.

| Tasks | Examples / run | Eligible train pool | Approx. pool passes |
|---:|---:|---:|---:|
| 1 | 1,279,904 | 490,000 | 2.612 |
| 2 | 1,279,968 | 980,000 | 1.306 |
| 4 | 1,280,000 | 1,960,000 | 0.653 |
| 8 | 1,280,000 | 3,920,000 | 0.327 |
| 16 | 1,280,000 | 7,840,000 | 0.163 |

Totals across the 30 runs:

```text
38,399,232 example exposures
807,897,938 supervised target tokens (including the EOS for every example)
600,000 optimizer steps
```

The average exposure per task therefore falls from approximately 1.28M at k=1 to 80k at k=16. The supervised-token budgets are not identical either: the k=1 runs trained only on `power` contain approximately 43.5M supervised tokens, whereas the other task mixtures contain approximately 22–23M per run. This is a confounding factor that must be retained when interpreting the task-count curves.

### 8.4 Validation protocol

Every 1,000 steps, all 20 tasks—not only the current training tasks—are evaluated on shard 098. Each task uses at most five dynamic batches; the final validation for each run contains 2,924 examples and 57,953 supervised tokens.

The following metrics are saved:

- token-weighted negative log likelihood.
- supervised token accuracy.
- exact sequence accuracy.
- example and supervised-token counts.

`token_accuracy` is teacher-forced: every token sees the gold prefix, so copy, punctuation, and boundary tokens inflate the value.

`sequence_accuracy` requires every next-token argmax for the answer and EOS to be correct. Because the models passed strict causality tests, this all-token event is equivalent to greedy decoding the complete canonical target from the same prompt; however, a separate parser-aware decoding harness was not run in this round.

Shard 099 was not used for model selection or any model evaluation in this round.

## 9. Checkpoints, Resume, and Completion Criteria

Each run's `checkpoint.pt` contains:

- model state.
- optimizer state.
- scheduler state.
- AMP scaler state.
- Python, CPU Torch, and CUDA RNG states.
- epoch, batch offset, and global step.
- per-task examples, tokens, and accumulated loss.
- complete `TrainConfig`.
- training/validation manifest fingerprints.
- last validation metrics.

Checkpoints are written through a temporary file followed by atomic replacement. Before resuming, the training configuration and data fingerprints are compared strictly; silently resuming with a different task set, shards, or hyperparameters is not permitted.

A run generates `completed.json` only after step 20,000 is complete and the final checkpoint has been written and hashed. The marker stores the checkpoint SHA-256, configuration hashes, task accounting, and validation metrics. An arbitrary invalid marker cannot be treated as evidence of completion.

A CUDA resume issue was discovered and corrected during training: when a checkpoint was loaded with `map_location=device`, the saved CUDA RNG states were mapped to CUDA tensors. Before restoration, they had to be converted uniformly back to CPU `uint8` tensors and then passed to `torch.cuda.set_rng_state_all`. After the fix, a formal run successfully resumed from its step-7,000 checkpoint and continued to completion.

## 10. Actual GPU Execution

### 10.1 Hardware/software

```text
GPU: NVIDIA GeForce RTX 5070, 12,227 MiB
Driver: 610.88
Python: 3.13.13
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
cuDNN: 9.19
bf16 support: true
```

### 10.2 Pilots

Before the formal matrix, separate 100-step, 16-task Transformer and MLP pilots were run to validate:

- CUDA forward/backward.
- bf16 AMP.
- dynamic batching.
- validation.
- checkpoint/marker.
- resume.
- VRAM usage and throughput.

Pilot artifacts are located at:

```text
runs/pilots/transformer-16task-100step-v2/
runs/pilots/mlp-16task-100step-v2/
```

### 10.3 Formal orchestration

The initial sequential controller was stopped cleanly after the first formal run reached step 7,000 in order to improve single-GPU utilization. After the CUDA RNG resume fix was completed and validated, the matrix was split into two non-overlapping run-ID queues. Each controller managed only one run from its own queue at a time, so the two controllers never wrote to the same directory concurrently.

The two queues shared one RTX 5070:

- Observed combined VRAM was typically approximately 3.5–4.2 GB.
- Observed GPU utilization was typically approximately 60–80%.
- No OOM occurred.
- No duplicate controller managed the same run.
- Runs with completed markers were skipped.
- Interrupted runs resumed automatically from their exact checkpoints.

The first formal completion marker was written at 01:31:32 and the last at 05:12:24, an interval of approximately 3 hours and 41 minutes. Both controllers ultimately exited normally.

### 10.4 Final artifacts

```text
runs/henry-permutation/<run-id>/checkpoint.pt
runs/henry-permutation/<run-id>/completed.json
```

- Transformer checkpoints: approximately 41.64 MB × 15.
- MLP checkpoints: approximately 35.20 MB × 15.
- The 30 formal checkpoints total 1,152,516,936 bytes.

`runs/` and production `data/` are excluded by `.gitignore` and therefore cannot be pushed to GitHub accidentally.

## 11. Preliminary Generalization Results

The cleanest cross-task-count comparison uses the four fixed holdouts that
were excluded from training for every model:

```text
to_reduced_word, compose, parity, to_lehmer
```

The following table reports the task-macro mean ± sample standard deviation across three seeds:

| Architecture | Trained tasks | Holdout token accuracy | Holdout exact-sequence accuracy |
|---|---:|---:|---:|
| Transformer | 1 | 19.18 ± 1.02% | 0.23 ± 0.00% |
| Transformer | 2 | 25.71 ± 1.36% | 0.88 ± 0.58% |
| Transformer | 4 | 30.89 ± 0.29% | 0.00 ± 0.00% |
| Transformer | 8 | **43.75 ± 0.22%** | 0.39 ± 0.13% |
| Transformer | 16 | 41.03 ± 1.94% | 0.31 ± 0.13% |
| MLP | 1 | 23.08 ± 0.38% | 0.00 ± 0.00% |
| MLP | 2 | 35.14 ± 2.71% | 1.17 ± 0.13% |
| MLP | 4 | 40.99 ± 2.75% | **3.12 ± 1.80%** |
| MLP | 8 | **49.05 ± 0.78%** | 1.06 ± 0.53% |
| MLP | 16 | 43.80 ± 0.56% | 0.83 ± 1.04% |

Observations:

1. Holdout token accuracy rises substantially from 1 to 8 training tasks, then declines slightly at 16 tasks.
2. The MLP has higher holdout token accuracy than the Transformer at every subset size.
3. This token-level transfer does not translate into reliable complete answers: the highest macro exact accuracy is only 3.12%.
4. Exact accuracy for `to_reduced_word` and `to_lehmer` is 0% under every condition.
5. The best three-seed exact mean for `compose` is 1.54% for the Transformer and 2.16% for the MLP; for `parity`, the corresponding values are 2.92% and 12.50%.

The most defensible current conclusion is:

> Increasing training-task diversity improves prefix-conditioned token transfer, with a peak under the 8-task condition, but does not produce reliable hard zero-shot complete-answer generalization.

A key design limitation is that the four holdouts use opaque task tokens. These tokens never appear as input tokens or correct targets in base-training sequences, so their operation semantics are not grounded, even though they remain part of the 163-way vocabulary and receive gradients as nontarget classes. The models were never taught the meaning of these tasks, making hard zero-shot task identification itself underdetermined. Henry's proposed few-shot adaptation and linear probing are more informative than this hard zero-shot metric.

Complete seen, pool-unseen, and holdout tables are provided in [results/v2/README.md](results/v2/README.md).

The per-holdout k=16 results further demonstrate that token accuracy must not be conflated with successful mathematical solution:

| Holdout task | Transformer token / exact | MLP token / exact |
|---|---:|---:|
| Reduced word | 16.03 ± 1.48% / 0.00 ± 0.00% | 18.54 ± 4.99% / 0.00 ± 0.00% |
| Composition | 61.30 ± 0.63% / 1.23 ± 0.53% | 60.06 ± 0.54% / 0.62 ± 0.53% |
| Parity | 40.00 ± 9.85% / 0.00 ± 0.00% | 51.35 ± 2.35% / 2.71 ± 4.69% |
| Lehmer code | 46.80 ± 3.48% / 0.00 ± 0.00% | 45.27 ± 8.17% / 0.00 ± 0.00% |

For example, composition achieves approximately 60% token accuracy but only about 1% exact accuracy, showing that correct local formatting or copying patterns do not imply a correct full operation. The nonzero mean for MLP k=16 parity is driven primarily by one seed, while the other two seeds achieve 0, so this result cannot be described as stable generalization.

## 12. Final Verification and Audit

### 12.1 Dataset verifier

```text
full=true
ok=true
record_count=10,000,000
shard_count=100
20 tasks × 500,000 records
```

### 12.2 Formal checkpoint audit

Final read-only audit results:

```text
run_count=30
passed_count=30
incomplete_count=0
failed_count=0
global issues=[]
partial artifacts=[]
```

The auditor does more than check that files exist. It:

- Reconstructs the complete expected `TrainConfig` from the frozen TOML and launch command.
- Validates experiment/manifest/checkpoint SHA values, manifest schema, frozen shard ranges, shard-file existence, and byte sizes; content SHA values and per-record mathematical recomputation for data shards are handled by the full dataset verifier described above.
- Safely reads checkpoints with `weights_only=True`.
- Instantiates the model under the expected architecture and strictly checks keys, shapes, and dtypes.
- Checks optimizer, scheduler, scaler, RNG, state, and validation schemas.
- Recursively checks for NaN/Inf values and impossible negative statistics.
- Compares marker accounting/validation against the checkpoint.
- Rejects symlinks, path escapes, and residual `.tmp/.partial/.part` files.
- Validates train/validation/test manifests and frozen shard ranges.

### 12.3 Tests

The repository contains 171 tests, all of which pass. Coverage includes:

- All 23 task definitions supported across v2 and v3 (20 per protocol).
- Passage Math grammar and canonical encodings.
- Deterministic generation, balance, resume, and corruption handling.
- Full ground-truth recomputation.
- Split views.
- Transformer/MLP causality, padding, gradients, serialization, and CUDA.
- Streaming loader, answer-only collator, training, resume, and markers.
- Experiment matrix.
- Adversarial completion audit.
- Audited validation-result export and one-time test evaluation.

The only warnings are deprecation warnings about using `fork()` from a multithreaded Python 3.13 process; they are neither test failures nor training-numerics issues.

## 13. Reproduction from Scratch

Run all commands below from the repository root.

### 13.1 Environment

```bash
git clone https://github.com/XuanyuYang223/neurips.git
cd neurips

# Python >=3.11 is recommended because the orchestration code uses tomllib.
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e '.[test,train]'
python -m pytest -q
```

The commands below target the current `main` branch. The historical audited
post-training baseline is commit
`32ff22a2e77acdf1d18b634ed431e54d3c1341f0`; its generator predates the
`--schema-version` option and therefore fixes v2 implicitly.

`pyproject.toml` currently declares Python ≥3.10, but `experiments.py` and `audit.py` import the standard-library `tomllib` directly; without installing a backport, Python ≥3.11 should be used in practice. All frozen paths are relative to the repository root, and the following commands should be run from that directory.

### 13.2 Generate and verify the 10M-record dataset

```bash
python -m neurips_permutations.generate \
  --count 10000000 \
  --max-entries 30 \
  --base 100 \
  --seed 20260830 \
  --shard-size 100000 \
  --workers 20 \
  --schema-version permutation-20/v2 \
  --output-dir data/permutation-10m-v2

python -m neurips_permutations.verify \
  data/permutation-10m-v2/manifest.json \
  --full \
  --workers 20

python -m neurips_permutations.splits \
  data/permutation-10m-v2/manifest.json \
  --train-shards 98 \
  --validation-shards 1 \
  --test-shards 1
```

The formal TOML intentionally points both training and validation to the parent manifest, then selects data using shard indices `000-097` and `098`. Do not replace these paths arbitrarily with split-manifest paths, because doing so changes the configuration hash and strict audit.

Current split-manifest SHA-256 values:

```text
train       76e682a8afb217350fbe4454eb473593f2cf53850254f826697faf6fa0349de3
validation  6bdc14e4363c2b8a0d74d389543d5260ef28597537cb578e8c37b4a0284693ef
test        9f9822b0dbac51af8c40d57fa5df12237ba1893c788582f5c1898f5ca33ed2da
```

### 13.3 Inspect the experiment matrix

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --plan

python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run \
  --dry-run
```

### 13.4 Train all 30 runs

The safest reproduction procedure uses a single controller to run the matrix sequentially:

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run
```

When the same command is executed again:

- Runs with a valid `completed.json` are skipped.
- Incomplete runs resume automatically from `checkpoint.pt`.
- A configuration or manifest hash mismatch fails immediately instead of mixing experiments.

The current runner performs single-process, single-GPU training and is not DDP. Do not switch directly to `torchrun` and still treat the results as belonging to the same frozen protocol. A formal reproduction should expose only one bf16-capable GPU, for example with `export CUDA_VISIBLE_DEVICES=0`.

One or more exact, non-overlapping run IDs may also be used:

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --run \
  --only transformer-tasks16-seed17
```

Do not allow two controllers to execute the same run ID concurrently, because the run directory has no cross-process lock.

Each run retains only one rolling `checkpoint.pt`; historical checkpoints from every 1,000 steps are not all preserved. These artifacts therefore support recovery and final auditing, but they cannot be used retrospectively to reconstruct the full learning curve or select a different best intermediate checkpoint.

### 13.5 Status and strict audit

```bash
python -m neurips_permutations.experiments \
  --config configs/henry_permutation.toml \
  --status

python -m neurips_permutations.audit \
  --config configs/henry_permutation.toml
```

## 14. Repository and Artifact Policy

The public GitHub repository stores:

- source code.
- frozen configs.
- tests and CI.
- protocol, experiment, and results documents.

GitHub does not store:

- 1.29 GB v2 production dataset.
- 1,139,175,228-byte v3 production dataset.
- 1.15 GB v2 formal checkpoints and 1,844,025,024 bytes of v3 checkpoints.
- pilot/checkpoint runtime directories.

Local paths:

```text
/home/yangx/neurips/data/permutation-10m-v2
/home/yangx/neurips/data/permutation-10m-v3
/home/yangx/neurips/runs/henry-permutation
/home/yangx/neurips/runs/henry-permutation-v3
```

To share these artifacts, use object storage, dataset hosting, GitHub Release assets, or a dedicated model registry rather than ordinary Git blobs.

## 15. Known Limitations and Next Steps

The `permutation-20/v3` data, 30-run nested matrix, 18-run E4/S4/A4 category matrix, strict audits, one-time independent zero-shot test evaluation, 144-run Henry-style 20-shot adaptation study, layerwise CKA, frozen category-model linear probes, and four-representation transfer grid are complete. The 30 old v2 models remain baseline/appendix material; they were not deleted or relabeled as v3 results. [results/v3/README.md](results/v3/README.md) reports the behavioral zero-shot result, [results/v3/fewshot/README.md](results/v3/fewshot/README.md) reports few-shot adaptation, [results/v3/linear-probing/category/README.md](results/v3/linear-probing/category/README.md) reports the category probe analysis, and [results/representation-transfer/README.md](results/representation-transfer/README.md) reports cross-representation/task transfer.

1. **Few-shot generalization**: The primary 20-shot protocol in [configs/henry_permutation_fewshot.toml](configs/henry_permutation_fewshot.toml) is complete. It adapted every nested base model separately to every fixed holdout at low learning rate. The post-hoc nested 5/20/100-shot curve is also complete and does not show a consistent support-size improvement.
2. **Random-init baseline**: The completed follow-up includes a paired random-initialization model for every architecture, task, and seed, using the same 20 support examples and number of updates. It uses the established from-scratch learning rate rather than the pretrained fine-tuning rate.
3. **Linear probes**: The completed Property32 and category-model probes extract layerwise hidden states at `<ONE_END>` before the task token, avoiding answer leakage. Alternative probe types and causal interventions remain optional extensions.
4. **Representation geometry**: Layerwise linear CKA and the controlled task-relation analyses are complete. SVCCA, Procrustes alignment, effective rank, and clustering remain optional extensions.
5. **Multiple task subsets**: The original R0/R1/R2 error bars jointly vary model seed and task partition. The completed fixed-seed R0/R3/R4 extension isolates balanced task-subset sensitivity at seed 17: mean final-layer CKA has Spearman rho 0.90 with `k`, but task-subset sample SD remains 0.11--0.18 and the mean still peaks at `k=8` rather than increasing strictly. More than three partitions would be needed for a precise population-level variance estimate.
6. **Fixed-budget interpretation**: As task count increases, exposure per task falls from approximately 1.28M to 80k. The current design changes diversity and per-task data simultaneously, so the decline at 16 tasks cannot be attributed directly to interference.
7. **Metric granularity**: Token accuracy includes delimiters, copied tokens, and EOS. Validation diagnostics use only 47–160 examples per task, while the final test uses 5,000 per task per model; per-task exact accuracy remains the primary complete-answer metric.
8. **Distribution scope**: Train, validation, and test are independent in-distribution shards with `n=2–30`. A post-hoc evaluation on `n=31–40` is complete and shows a large exact-accuracy collapse. It simultaneously shifts length and number-token frequencies, so it is evidence of extrapolation failure rather than a diagnosis of its cause. Other combinatorial distribution shifts remain untested.
9. **Architecture matching**: The Transformer has 3.463M registered parameters and the MLP has 2.930M, so the former has approximately 18.2% more. However, 1,047,552 strict upper-triangular token-mixing parameters in the MLP are masked and do not participate in the forward pass, so the nominal count should not be interpreted as active-capacity matching either. The two architectures did not each undergo complete hyperparameter tuning, so the results support descriptive comparisons, not strong causal claims.
10. **Statistical scope**: There are only three seeds, with no task-level bootstrap or significance test. Nonzero values such as MLP parity can be driven by a single seed.
11. **Opaque holdout tokens**: Hard zero-shot inference cannot recover the meaning of an unseen operation from the token itself. Shared semantics, task descriptions, cross-representation combinations, or few-shot supervision are needed.
12. **Determinism**: The seed, data order, and sharding are deterministic, but `torch.use_deterministic_algorithms` was not enabled; bitwise-identical weights are not guaranteed across GPU, CUDA, or PyTorch versions.
13. **Environment provenance**: The checkpoint/marker does not embed the Git commit or Python, Torch, CUDA, or driver versions. This document records the environment used here, but future protocols should write these values into the marker.
14. **4×8 representation grid**: This extension is complete. Three joint Transformers were trained on the one-line row and descents column (11 cells) and evaluated once on all 32 cells. Exact accuracy was `54.31 ± 0.96%` on trained cells and `30.61 ± 2.64%` on the 21 held-out cells; the held-out advantage over cell-specific constant-answer majority baselines was `+11.46 ± 2.64` percentage points. The complete matrix is in [results/representation-transfer/README.md](results/representation-transfer/README.md).

## 16. Key File Index

- [README.md](README.md): quick start.
- [PROTOCOL.md](PROTOCOL.md): 20-task mathematics and data protocol.
- [EXPERIMENTS.md](EXPERIMENTS.md): overview of Henry's nested matrix.
- [results/v2/README.md](results/v2/README.md): v2 validation and generalization tables.
- [results/v3/README.md](results/v3/README.md): completed 48-model v3 training and independent test results.
- [results/v3/fewshot/README.md](results/v3/fewshot/README.md): completed Henry-style 20-shot adaptation results.
- [results/representation-transfer/README.md](results/representation-transfer/README.md): completed four-representation by eight-task transfer matrix.
- [results/property32-zero-overlap/subset-replicates/README.md](results/property32-zero-overlap/subset-replicates/README.md): fixed-seed task-subset sensitivity for behavioral transfer and CKA.
- [configs/henry_permutation.toml](configs/henry_permutation.toml): frozen experiment configuration.
- [configs/henry_permutation_revised.toml](configs/henry_permutation_revised.toml): frozen v3 launch design after Henry's feedback.
- [configs/henry_permutation_fewshot.toml](configs/henry_permutation_fewshot.toml): frozen 20-shot follow-up.
- [generate.py](src/neurips_permutations/generate.py): data generation.
- [verify.py](src/neurips_permutations/verify.py): full data verification.
- [passage.py](src/neurips_permutations/passage.py): tokenizer and Passage Math grammar.
- [models.py](src/neurips_permutations/models.py): Transformer and causal MLP.
- [training.py](src/neurips_permutations/training.py): streaming training and checkpoint/resume.
- [experiments.py](src/neurips_permutations/experiments.py): 30-run orchestration.
- [audit.py](src/neurips_permutations/audit.py): strict completion audit.
- [evaluate.py](src/neurips_permutations/evaluate.py): one-time full test evaluation.
- [results.py](src/neurips_permutations/results.py): audited validation/test result export.
- [fewshot.py](src/neurips_permutations/fewshot.py): support selection, adaptation, audit, and test evaluation.
- [fewshot_results.py](src/neurips_permutations/fewshot_results.py): few-shot result aggregation and paired gains.

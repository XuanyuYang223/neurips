# Four-Representation Transfer Protocol

This experiment directly implements the supplied 4 × 8 Passage Math design.
It asks whether one jointly trained Transformer can transfer a task across
permutation representations.

## Grid

The input representations are one-line notation, canonical cycle notation,
Lehmer code, and inversion vector. The tasks are Coxeter length (inversion
count), parity, peaks, exceedances, fixed points, descents, recoils, and LIS
length. This yields 32 representation-task combinations.

Each model is trained on the union of:

- all eight tasks in one-line notation; and
- descents in cycle, Lehmer, and inversion-vector notation.

The overlap cell `one_line:descents` is counted once, so the model sees 11
cells. The other 21 cells receive no gradient updates. All 32 cells are
evaluated. This is a single joint model per seed, not 11 independently trained
specialists.

## Data and controls

The data are deterministically derived from the descents records in the fully
verified 16-million-row Property32 corpus. Each source permutation is expanded
over the relevant grid, so all representation-task cells use exactly matched
permutations. The train, validation, and test source splits remain separate.
Every training cell contains 490,000 records; every validation and test cell
contains 5,000 records. The derived training corpus therefore has 5,390,000
rows, and each evaluation split has 160,000 rows.

All cells use the same base-100 tokenizer and scalar answer grammar. Only the
representation boundary and body tokens change. Three standard decoder-only
Transformers use the frozen seeds 17, 42, and 314159. Each has four pre-LN
layers, eight heads, hidden width 256, feed-forward width 1,024, dropout 0.1,
and tied input/output embeddings. Optimization matches the other permutation
experiments: 20,000 AdamW updates, learning rate 3e-4 with 1,000 warmup steps
and cosine decay, effective batch size 64, and bf16 AMP.

## Primary result

The primary comparison is task-macro exact-sequence accuracy on the 11 trained
cells versus the 21 held-out cells, computed within each seed and then reported
as mean ± sample standard deviation over the three seeds. The complete 4 × 8
matrix is also reported. Token accuracy is secondary because it is
teacher-forced and can be inflated by shared formatting tokens.

The 21 held-out cells test cross-representation behavioral transfer. They do
not by themselves prove that the internal representations are identical;
hidden-state CKA or probing would be a separate mechanistic analysis.

## Commands

```bash
python -m neurips_permutations.representation_transfer prepare --workers 8
python -m neurips_permutations.representation_transfer verify --full
python -m neurips_permutations.representation_transfer plan
python -m neurips_permutations.representation_transfer run
python -m neurips_permutations.representation_transfer audit
python -m neurips_permutations.representation_transfer test --device cuda
python -m neurips_permutations.representation_transfer report
```

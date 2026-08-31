# NeurIPS permutation multitask data

This repository contains the completed `permutation-20/v2` baseline and the
revised `permutation-20/v3` main-study design for the permutation half of the
multitask-generalization study. The completed v2 baseline covers exactly 20
tasks:

- 4 encodings/translations;
- 9 statistics/properties;
- 7 algebraic operations/comparisons.

The v2 production run wrote **10,000,000 final Passage Math sequences**,
balanced exactly across the 20 tasks (500,000 per task). A record is one model
sequence with one task target, matching the supplied Passage Math convention.

## Henry revision: v3 main study

Following Henry Kvinge's feedback, the paper's revised main suite keeps the
standard small pre-LN decoder-only Transformer and does not attempt to match
the original PermuFormer architecture.  Henry suggested excluding the three
difficult algebraic tasks and comparing representations learned from different
task categories.  The choice of `peaks`, `exceedances`, and `recoils` as the
three replacements is a project decision made to preserve 20 balanced tasks:

| v2 task removed | v3 task added | v3 records |
|---|---|---:|
| `power` | `peaks` | 500,000 |
| `conjugate` | `exceedances` | 500,000 |
| `commutator` | `recoils` | 500,000 |

The v3 dataset is complete under `data/permutation-10m-v3`: `20 × 500,000 =
10,000,000` records in 100 gzip shards, totaling 1,139,175,228 compressed
bytes.  Generation took 43.76 seconds and full record-by-record mathematical
and encoding verification took 34.59 seconds.  The parent manifest SHA-256 is
`b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f`.
The checked-in [public manifest snapshot](manifests/permutation-10m-v3.json) is
byte-identical to the local source manifest; the verification run is recorded
in the [public full-verification report](manifests/permutation-10m-v3-verification.json).

| Split | Records | Manifest SHA-256 |
|---|---:|---|
| Train, shards `000-097` | 9,800,000 | `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa` |
| Validation, shard `098` | 100,000 | `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068` |
| Test, shard `099` | 100,000 | `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b` |

V2 data and all 30 completed v2 models are retained as the baseline; they are
not relabelled as revised results.

The project's operationalization of Henry's representation-comparison idea is
a task-count-matched E4/S4/A4 design: four encoding tasks, four statistics
tasks, and four algebra tasks.  Details and exact task lists are in
[EXPERIMENTS.md](EXPERIMENTS.md) and
[`configs/henry_permutation_revised.toml`](configs/henry_permutation_revised.toml).

**Status:** the v3 data is generated and full-verified, but no revised v3 model
has been trained yet.  The accuracy matrices below are exclusively completed
v2 baseline results.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest

# Small, deterministic smoke corpus.
permutation-generate --count 2000 --output-dir data/smoke --workers 1
permutation-verify data/smoke/manifest.json --full

# Full corpus: 100 gzip shards of 100,000 records each.
permutation-generate \
  --count 10000000 \
  --max-entries 30 \
  --base 100 \
  --seed 20260830 \
  --shard-size 100000 \
  --workers 20 \
  --schema-version permutation-20/v3 \
  --output-dir data/permutation-10m-v3
permutation-verify data/permutation-10m-v3/manifest.json --full --workers 20
permutation-split data/permutation-10m-v3/manifest.json
```

Generation is deterministic, streaming, parallel, and resumable. Completed
shards are reused only after their checksum and record count are verified.
The manifest records the seed, protocol version, task counts, shard hashes, and
byte sizes.

## Important interpretation

The mathematical objects are standard permutations of `{1, ..., n}` with
`2 <= n <= 30`. Thus the largest permutation entry is 30. The requested
"maximum number 100" is implemented as the supplied **base-100 tokenizer**:
`00` through `99` are atomic tokens, while values at least 100 are encoded
canonically between `<NUM_START>` and `<NUM_END>`.

The generated multi-gigabyte data shards are ignored by Git and should be
stored as release artifacts, object-store datasets, or local research artifacts
rather than committed to ordinary Git history.  The repository publishes the
complete v3 [manifest snapshot](manifests/permutation-10m-v3.json) and
[verification summary](manifests/permutation-10m-v3-verification.json), not a
sample of the underlying records.

See [PROTOCOL.md](PROTOCOL.md) for exact definitions, composition conventions,
canonical output rules, and the extended Passage Math grammar.

## Experiment documentation

- [TRAINING_PROCESS.md](TRAINING_PROCESS.md): complete data-generation,
  encoding, architecture, training, recovery, validation, audit, and
  generalization record.
- [EXPERIMENTS.md](EXPERIMENTS.md): completed v2 baseline, revised v3 nested
  matrix, and matched E4/S4/A4 category design.
- [TRAINING_RESULTS.md](TRAINING_RESULTS.md): final validation tables and
  generalization interpretation.
- [MODEL_TASK_ACCURACIES.csv](MODEL_TASK_ACCURACIES.csv): all 600 unaveraged
  model-by-task token and exact-sequence accuracy rows in filterable form.

## Every v2 baseline model, every v2 task: validation accuracy

The following matrices contain the unaveraged result for every one of the
30 completed v2 models on every one of the v2 tasks. Each cell is one specific
model-task result from validation shard 098; no task or seed averaging is
performed. Values are percentages.

The four fixed holdouts are `RWORD`, `COMP`, `PAR`, `LEHM`. For a
model labelled `k=N`, the first N tasks among the first 16 columns were
used for training; the remaining first-16 tasks are pool-unseen. Test shard
099 remains untouched.

| Abbreviation | Task | Abbreviation | Task |
|---|---|---|---|
| `PWR` | `power` | `LIS` | `lis_length` |
| `FIX` | `fixed_points` | `IVEC` | `to_inversion_vector` |
| `PAV` | `pattern_avoidance` | `LDS` | `lds_length` |
| `RMS` | `right_multiply_simple` | `CONJ` | `conjugate` |
| `CYC` | `to_cycle` | `DESC` | `descents` |
| `COMM` | `commutator` | `LEN` | `length` |
| `RSK` | `rsk_shape` | `INV` | `inverse` |
| `CTYPE` | `cycle_type` | `BRU` | `bruhat_leq` |
| `RWORD` | `to_reduced_word` | `COMP` | `compose` |
| `PAR` | `parity` | `LEHM` | `to_lehmer` |

<details open>
<summary>Transformer: exact-sequence accuracy</summary>

| Model | PWR | LIS | FIX | IVEC | PAV | LDS | RMS | CONJ | CYC | DESC | COMM | LEN | RSK | INV | CTYPE | BRU | RWORD | COMP | PAR | LEHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k1-s17 | 23.72 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.92 | 0.00 | 0.00 | 4.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 0.00 | 0.00 |
| k1-s42 | 25.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.25 | 0.00 | 0.00 | 0.00 | 0.93 | 0.00 | 0.00 |
| k1-s314159 | 27.56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.64 | 2.75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 0.00 | 0.00 |
| k2-s17 | 15.38 | 86.25 | 3.75 | 0.00 | 0.00 | 22.50 | 0.00 | 0.92 | 0.00 | 11.25 | 4.55 | 3.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 4.38 | 0.00 |
| k2-s42 | 16.67 | 91.25 | 3.75 | 0.00 | 0.00 | 17.50 | 0.00 | 0.92 | 0.00 | 6.88 | 4.55 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 0.00 | 0.00 |
| k2-s314159 | 15.38 | 91.88 | 3.75 | 0.00 | 0.00 | 18.75 | 0.00 | 0.00 | 0.00 | 9.38 | 0.00 | 3.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.38 | 0.00 |
| k4-s17 | 14.10 | 77.50 | 96.88 | 48.12 | 0.00 | 1.25 | 0.00 | 0.00 | 0.00 | 1.88 | 0.00 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k4-s42 | 14.74 | 74.38 | 100.00 | 37.50 | 0.00 | 0.00 | 5.13 | 0.00 | 0.00 | 0.00 | 0.00 | 1.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k4-s314159 | 12.82 | 66.88 | 99.38 | 35.00 | 0.00 | 0.00 | 0.00 | 0.92 | 0.00 | 0.00 | 0.91 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k8-s17 | 14.10 | 70.62 | 99.38 | 23.12 | 99.38 | 65.00 | 26.28 | 7.34 | 0.00 | 0.00 | 1.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.85 | 0.00 | 0.00 |
| k8-s42 | 13.46 | 70.00 | 97.50 | 22.50 | 98.75 | 65.00 | 28.21 | 7.34 | 0.00 | 0.00 | 1.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 0.00 | 0.00 |
| k8-s314159 | 16.03 | 72.50 | 98.12 | 25.00 | 98.75 | 60.62 | 30.13 | 10.09 | 0.00 | 0.00 | 1.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.85 | 0.00 | 0.00 |
| k16-s17 | 7.05 | 68.12 | 88.75 | 20.00 | 86.25 | 63.12 | 14.10 | 8.26 | 12.50 | 55.00 | 6.36 | 38.75 | 23.75 | 14.37 | 19.38 | 98.10 | 0.00 | 0.93 | 0.00 | 0.00 |
| k16-s42 | 8.33 | 68.12 | 81.88 | 21.25 | 94.38 | 61.25 | 14.10 | 5.50 | 10.62 | 47.50 | 5.45 | 40.00 | 25.62 | 17.50 | 18.12 | 98.73 | 0.00 | 0.93 | 0.00 | 0.00 |
| k16-s314159 | 7.05 | 70.62 | 78.75 | 20.00 | 89.38 | 62.50 | 16.03 | 5.50 | 11.88 | 51.25 | 6.36 | 39.38 | 23.75 | 15.00 | 20.00 | 98.73 | 0.00 | 1.85 | 0.00 | 0.00 |

</details>

<details open>
<summary>MLP: exact-sequence accuracy</summary>

| Model | PWR | LIS | FIX | IVEC | PAV | LDS | RMS | CONJ | CYC | DESC | COMM | LEN | RSK | INV | CTYPE | BRU | RWORD | COMP | PAR | LEHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k1-s17 | 16.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k1-s42 | 17.31 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k1-s314159 | 17.95 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| k2-s17 | 15.38 | 56.88 | 6.88 | 0.00 | 0.00 | 25.00 | 1.28 | 0.00 | 0.00 | 15.62 | 0.00 | 4.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.38 | 0.00 |
| k2-s42 | 15.38 | 53.75 | 6.88 | 0.00 | 0.00 | 23.75 | 1.28 | 0.92 | 0.00 | 17.50 | 0.91 | 4.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.38 | 0.00 |
| k2-s314159 | 15.38 | 54.37 | 6.25 | 0.00 | 0.00 | 21.88 | 0.64 | 0.92 | 0.00 | 16.88 | 4.55 | 4.38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 4.38 | 0.00 |
| k4-s17 | 12.82 | 56.25 | 67.50 | 17.50 | 0.00 | 5.62 | 5.13 | 0.00 | 0.00 | 5.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 15.00 | 0.00 |
| k4-s42 | 14.74 | 56.88 | 73.75 | 14.37 | 0.00 | 1.88 | 0.00 | 0.00 | 0.00 | 3.12 | 0.00 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 4.38 | 0.00 |
| k4-s314159 | 13.46 | 58.13 | 71.25 | 15.62 | 0.00 | 4.38 | 0.00 | 0.00 | 0.00 | 6.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 18.12 | 0.00 |
| k8-s17 | 8.33 | 52.50 | 60.00 | 15.00 | 88.75 | 53.12 | 12.82 | 5.50 | 0.00 | 3.12 | 1.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.85 | 0.00 | 0.00 |
| k8-s42 | 7.69 | 54.37 | 56.25 | 14.37 | 80.62 | 51.88 | 13.46 | 6.42 | 0.00 | 8.12 | 1.82 | 0.62 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 3.70 | 1.25 | 0.00 |
| k8-s314159 | 7.69 | 54.37 | 57.50 | 15.00 | 90.62 | 53.12 | 7.69 | 6.42 | 0.00 | 5.00 | 1.82 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.93 | 5.00 | 0.00 |
| k16-s17 | 7.05 | 51.88 | 46.25 | 11.88 | 65.62 | 51.88 | 8.97 | 5.50 | 6.25 | 46.25 | 5.45 | 32.50 | 15.00 | 8.12 | 10.00 | 89.24 | 0.00 | 0.00 | 8.12 | 0.00 |
| k16-s42 | 1.92 | 51.25 | 46.25 | 12.50 | 63.75 | 51.25 | 6.41 | 3.67 | 6.25 | 43.75 | 6.36 | 29.38 | 15.00 | 6.25 | 9.38 | 86.71 | 0.00 | 0.93 | 0.00 | 0.00 |
| k16-s314159 | 3.85 | 51.88 | 45.62 | 13.12 | 58.13 | 50.62 | 7.05 | 6.42 | 7.50 | 47.50 | 5.45 | 29.38 | 11.25 | 6.88 | 9.38 | 87.34 | 0.00 | 0.93 | 0.00 | 0.00 |

</details>

<details>
<summary>Transformer: teacher-forced token accuracy</summary>

| Model | PWR | LIS | FIX | IVEC | PAV | LDS | RMS | CONJ | CYC | DESC | COMM | LEN | RSK | INV | CTYPE | BRU | RWORD | COMP | PAR | LEHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k1-s17 | 74.64 | 0.00 | 0.31 | 39.07 | 0.62 | 0.00 | 89.97 | 17.10 | 43.33 | 0.00 | 16.46 | 0.00 | 36.64 | 68.43 | 28.42 | 0.32 | 16.84 | 16.07 | 0.00 | 39.09 |
| k1-s42 | 74.83 | 0.00 | 0.00 | 39.45 | 1.25 | 0.00 | 88.71 | 22.76 | 42.91 | 0.00 | 22.43 | 0.00 | 36.93 | 64.98 | 28.87 | 1.27 | 17.49 | 22.17 | 0.00 | 39.29 |
| k1-s314159 | 74.68 | 0.00 | 0.00 | 39.88 | 0.31 | 0.00 | 88.30 | 21.94 | 43.59 | 0.00 | 21.51 | 0.21 | 37.85 | 72.70 | 28.87 | 0.63 | 18.99 | 20.61 | 0.00 | 39.58 |
| k2-s17 | 67.88 | 93.12 | 29.38 | 39.95 | 8.44 | 57.50 | 62.50 | 23.01 | 42.22 | 52.50 | 23.35 | 17.32 | 42.72 | 55.49 | 31.00 | 7.91 | 16.68 | 22.62 | 26.88 | 39.92 |
| k2-s42 | 67.15 | 95.62 | 23.75 | 37.28 | 0.00 | 54.69 | 58.81 | 28.25 | 41.00 | 47.50 | 26.62 | 8.87 | 37.32 | 54.25 | 28.42 | 0.00 | 16.20 | 27.33 | 15.94 | 37.07 |
| k2-s314159 | 67.36 | 95.94 | 27.50 | 44.10 | 13.12 | 54.06 | 64.46 | 23.13 | 44.77 | 49.69 | 21.51 | 13.40 | 44.74 | 54.58 | 35.27 | 14.87 | 17.54 | 22.52 | 21.88 | 43.95 |
| k4-s17 | 66.34 | 88.75 | 98.44 | 91.08 | 0.31 | 49.69 | 61.76 | 9.41 | 44.84 | 50.00 | 9.25 | 18.35 | 42.09 | 58.54 | 35.57 | 0.95 | 14.07 | 8.30 | 50.00 | 50.80 |
| k4-s42 | 66.75 | 87.19 | 100.00 | 85.41 | 0.00 | 47.81 | 54.67 | 9.55 | 43.33 | 48.44 | 8.94 | 15.88 | 41.51 | 50.88 | 35.05 | 0.00 | 11.68 | 8.32 | 50.00 | 52.67 |
| k4-s314159 | 67.13 | 83.44 | 99.69 | 83.67 | 2.50 | 49.69 | 55.56 | 10.93 | 41.64 | 47.19 | 11.29 | 16.49 | 37.70 | 50.75 | 29.53 | 2.85 | 11.57 | 10.89 | 50.00 | 52.41 |
| k8-s17 | 67.02 | 85.31 | 99.69 | 77.61 | 99.69 | 82.50 | 92.05 | 62.39 | 45.00 | 50.00 | 61.50 | 20.82 | 42.19 | 57.26 | 37.85 | 0.00 | 14.88 | 61.84 | 50.00 | 48.87 |
| k8-s42 | 67.67 | 85.00 | 98.75 | 75.39 | 99.38 | 82.50 | 92.67 | 63.18 | 44.25 | 49.06 | 61.79 | 13.40 | 37.13 | 57.33 | 29.90 | 0.00 | 15.53 | 61.60 | 50.00 | 48.26 |
| k8-s314159 | 68.56 | 86.25 | 99.06 | 78.01 | 99.38 | 80.31 | 94.81 | 62.65 | 44.74 | 50.00 | 61.90 | 20.62 | 36.98 | 57.49 | 29.82 | 0.00 | 15.31 | 61.44 | 50.00 | 47.24 |
| k16-s17 | 65.05 | 84.06 | 94.38 | 72.78 | 93.12 | 81.56 | 76.56 | 62.56 | 64.75 | 77.50 | 61.29 | 79.59 | 86.79 | 70.45 | 81.15 | 99.05 | 17.69 | 61.50 | 30.31 | 45.72 |
| k16-s42 | 65.27 | 84.06 | 90.94 | 73.42 | 97.19 | 80.62 | 76.92 | 61.97 | 64.63 | 73.75 | 61.22 | 80.00 | 86.84 | 71.58 | 78.87 | 99.37 | 14.84 | 60.60 | 50.00 | 43.99 |
| k16-s314159 | 64.33 | 85.31 | 89.38 | 72.63 | 94.69 | 81.25 | 77.00 | 62.31 | 64.56 | 75.62 | 61.16 | 79.38 | 87.17 | 70.34 | 78.79 | 99.37 | 15.57 | 61.81 | 39.69 | 50.69 |

</details>

<details>
<summary>MLP: teacher-forced token accuracy</summary>

| Model | PWR | LIS | FIX | IVEC | PAV | LDS | RMS | CONJ | CYC | DESC | COMM | LEN | RSK | INV | CTYPE | BRU | RWORD | COMP | PAR | LEHM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k1-s17 | 69.55 | 0.00 | 0.00 | 41.78 | 1.25 | 0.00 | 68.04 | 21.52 | 41.61 | 0.00 | 22.59 | 4.54 | 35.92 | 59.13 | 25.63 | 0.32 | 27.69 | 22.97 | 0.00 | 41.47 |
| k1-s42 | 69.50 | 0.00 | 0.00 | 42.62 | 0.62 | 0.00 | 68.27 | 23.80 | 43.05 | 0.00 | 24.61 | 3.30 | 34.28 | 58.06 | 26.29 | 0.32 | 24.03 | 24.92 | 0.00 | 41.94 |
| k1-s314159 | 69.27 | 0.00 | 0.00 | 40.10 | 0.62 | 0.00 | 67.44 | 22.76 | 42.63 | 0.00 | 22.82 | 2.89 | 33.61 | 56.99 | 25.48 | 0.32 | 28.53 | 24.34 | 0.31 | 40.76 |
| k2-s17 | 68.31 | 78.44 | 33.75 | 45.09 | 0.62 | 62.50 | 68.72 | 38.90 | 44.01 | 55.00 | 37.60 | 16.49 | 33.32 | 56.13 | 24.82 | 0.32 | 35.58 | 36.81 | 25.62 | 44.93 |
| k2-s42 | 68.37 | 76.88 | 53.44 | 41.52 | 15.31 | 61.88 | 67.69 | 23.80 | 46.75 | 58.75 | 24.61 | 23.09 | 37.99 | 55.68 | 36.30 | 9.81 | 30.68 | 25.69 | 52.19 | 41.41 |
| k2-s314159 | 68.48 | 77.19 | 37.19 | 39.07 | 0.62 | 60.31 | 67.46 | 26.45 | 45.10 | 55.00 | 25.73 | 17.53 | 35.54 | 54.76 | 29.31 | 1.27 | 29.31 | 26.69 | 33.75 | 38.96 |
| k4-s17 | 67.02 | 78.12 | 83.75 | 65.71 | 0.00 | 52.81 | 64.87 | 14.54 | 44.46 | 52.50 | 15.64 | 30.52 | 40.79 | 52.88 | 33.43 | 0.63 | 36.75 | 15.01 | 57.50 | 54.08 |
| k4-s42 | 67.69 | 78.44 | 86.88 | 64.90 | 13.12 | 50.94 | 58.46 | 14.87 | 41.71 | 51.56 | 16.72 | 21.86 | 42.48 | 53.29 | 39.18 | 7.59 | 29.84 | 16.46 | 52.19 | 54.79 |
| k4-s314159 | 67.04 | 79.06 | 85.62 | 65.14 | 5.00 | 52.19 | 59.74 | 23.89 | 43.64 | 53.12 | 24.63 | 21.65 | 43.35 | 53.77 | 40.87 | 1.58 | 36.66 | 25.63 | 59.06 | 53.89 |
| k8-s17 | 66.23 | 76.25 | 80.00 | 63.91 | 94.38 | 76.56 | 82.91 | 63.41 | 42.70 | 51.56 | 61.53 | 29.90 | 40.84 | 53.58 | 35.20 | 20.25 | 37.80 | 61.21 | 50.00 | 49.49 |
| k8-s42 | 66.47 | 77.19 | 78.12 | 64.17 | 90.31 | 75.94 | 83.59 | 63.75 | 41.77 | 54.06 | 62.08 | 31.96 | 36.16 | 57.19 | 31.81 | 16.77 | 38.73 | 61.68 | 50.31 | 46.79 |
| k8-s314159 | 66.36 | 77.19 | 78.75 | 64.19 | 95.31 | 76.56 | 79.08 | 63.10 | 42.43 | 52.50 | 62.16 | 20.00 | 40.26 | 56.33 | 31.59 | 17.09 | 25.48 | 61.87 | 52.50 | 52.80 |
| k16-s17 | 60.67 | 75.94 | 73.12 | 63.95 | 82.81 | 75.94 | 61.51 | 62.82 | 57.37 | 73.12 | 62.53 | 77.11 | 80.42 | 59.47 | 70.40 | 94.62 | 23.59 | 60.39 | 54.06 | 35.89 |
| k16-s42 | 58.06 | 75.62 | 73.12 | 64.02 | 81.88 | 75.62 | 59.33 | 61.46 | 54.67 | 71.88 | 60.80 | 76.49 | 80.62 | 57.83 | 70.10 | 93.35 | 13.61 | 59.43 | 50.00 | 50.85 |
| k16-s314159 | 59.54 | 75.94 | 72.81 | 63.84 | 79.06 | 75.31 | 59.37 | 61.30 | 55.84 | 73.75 | 61.06 | 75.88 | 79.65 | 58.83 | 68.70 | 93.67 | 18.40 | 60.36 | 50.00 | 49.06 |

</details>

Exact-sequence accuracy is the primary complete-answer correctness metric.
Token accuracy is teacher-forced and can be inflated by delimiters, copying,
and gold answer prefixes. The same 600 model-task rows, including task
status and evaluation counts, are available in
[`MODEL_TASK_ACCURACIES.csv`](MODEL_TASK_ACCURACIES.csv).

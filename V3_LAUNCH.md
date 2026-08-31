# v3 Launch Record

This record freezes the provenance and execution plan for the revised v3
study before any formal v3 model is trained. The completed v2 runs are a
separate baseline and will not be overwritten.

## Frozen provenance

- Implementation commit: `d1d163bf2b3209dc5b6cc61ac4396d84fa6e2613`
- Experiment configuration SHA-256:
  `dd75f31277e42f554ed681beda44bb53f2d4f65089fd9583540b9e645c4f1b40`
- Dataset protocol: `permutation-20/v3`
- Parent manifest SHA-256:
  `b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f`
- Train manifest SHA-256:
  `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa`
- Validation manifest SHA-256:
  `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068`
- Test manifest SHA-256:
  `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b`
- Python: 3.13.13
- PyTorch: 2.11.0+cu128
- CUDA runtime: 12.8
- cuDNN: 9.19
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB
- NVIDIA driver: 610.88
- Precision: bfloat16 automatic mixed precision

The 10,000,000-record parent dataset passed full record-by-record verification
immediately before launch. It contains 100 shards and exactly 500,000 records
for each of the 20 v3 tasks. The train, validation, and untouched test views
contain 9,800,000, 100,000, and 100,000 records, respectively.

## Formal matrices

The nested matrix contains 30 runs: two architectures, task counts
`1, 2, 4, 8, 16`, and seeds `17, 42, 314159`. It uses one frozen nested task
order and holds the final four tasks out of all gradient updates.

The category matrix contains 18 runs: two architectures, three seeds, and
three four-task conditions:

- E4: `to_cycle`, `to_lehmer`, `to_inversion_vector`, `to_reduced_word`;
- S4: `length`, `cycle_type`, `rsk_shape`, `pattern_avoidance`;
- A4: `inverse`, `compose`, `right_multiply_simple`, `bruhat_leq`.

S4 is a frozen representative subset of the 12 v3 statistics tasks; results
must not be described as covering every possible statistics-task selection.

Every run uses 20,000 optimizer updates. Nested runs use a micro-batch of 16
and four gradient-accumulation steps. To keep the category conditions at the
same effective batch size of 64 examples despite long reduced-word outputs,
E4 uses `4 x 16`, while S4 and A4 use `16 x 4`. This matches examples per
optimizer update, not supervised-token or FLOP budgets.

## Preflight evidence

The complete test suite passed with 155 tests. Both strict prelaunch audits
reported no failed runs and no global or manifest issues; all 48 expected runs
were correctly classified as incomplete. Four isolated 100-update GPU pilots
completed without NaN, Inf, OOM, checkpoint, or completion-marker failures:

| Pilot | Wall time |
|---|---:|
| Nested 16-task Transformer | 9.68 s |
| Nested 16-task MLP | 5.23 s |
| Category E4 Transformer (`4 x 16`) | 25.15 s |
| Category E4 MLP (`4 x 16`) | 10.55 s |

Pilot artifacts are stored under `runs/pilots/` and are never considered
formal matrix results.

## Execution and completion criteria

One nested controller and one category controller may run concurrently. Their
run identifiers and output directories are disjoint. A second controller for
the same matrix must not be launched because the runner has no cross-process
lock for duplicate run identifiers.

A matrix is complete only when every expected run has a valid
`completed.json` at global step 20,000 and passes the strict audit, including
configuration and data fingerprints, checkpoint SHA-256, checkpoint schema,
finite model and optimizer tensors, task accounting, and validation coverage.
The untouched test shard is reserved for one final evaluation after all model
selection and validation-based analysis are frozen.

## Interpretation limits

The nested study fixes total optimizer updates, so increasing task count
reduces per-task exposure. It uses one task order; the three seeds vary model
initialization and data order, not task selection. Validation shard 098 is
consulted during training and therefore is not the final test set. Token
accuracy is teacher-forced and includes structural tokens and EOS; exact
sequence accuracy is the primary operation-level metric. Category comparisons
match task count and examples per update but do not match target-token counts
or compute.

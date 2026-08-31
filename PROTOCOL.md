# Dataset protocol

Two immutable protocol versions are documented:

- `permutation-20/v2` is the completed baseline used by the existing 30
  models and the accuracy matrices in the README.
- `permutation-20/v3` is the revised main-study protocol motivated by Henry
  Kvinge's feedback.  Its dataset has been generated and fully verified.  It
  replaces three expensive algebraic targets while retaining 20 balanced
  tasks.  No v3 model result may be reported until a corresponding model
  completion marker exists.

## v3 revision

The primary input, number encoding, grammar, sampling range, and retained task
definitions are unchanged from v2.  Henry suggested removing the three
difficult v2 tasks and comparing task categories; this project chose the three
replacement properties below to preserve a balanced 20-task suite:

| v2 task removed | v3 task added | v3 definition |
|---|---|---|
| `power` | `peaks` | `#{i : 2 <= i <= n-1, pi_(i-1) < pi_i > pi_(i+1)}` |
| `conjugate` | `exceedances` | `#{i : 1 <= i <= n, pi(i) > i}` |
| `commutator` | `recoils` | `descent_count(pi^-1)`, equivalently `#{i : position(i) > position(i+1)}` |

All three answers are nonnegative scalar counts.  With `2 <= n <= 30`, each
answer is below 100 and therefore uses one atomic two-digit number token; for
example, zero is `00`, not `<NUM_START> 00 <NUM_END>`.

The canonical v3 task registry consists of:

```text
Encoding/translation (4):
  to_cycle, to_lehmer, to_inversion_vector, to_reduced_word

Statistics/properties (12):
  length, descents, fixed_points, parity, cycle_type, rsk_shape,
  lis_length, lds_length, pattern_avoidance, peaks, exceedances, recoils

Algebra/comparison (4):
  inverse, compose, right_multiply_simple, bruhat_leq
```

The completed v3 corpus contains 10,000,000 records in 100 gzip shards, exactly
500,000 per task.  In particular, `peaks`, `exceedances`, and `recoils` each
have 500,000 records.  It occupies 1,139,175,228 compressed bytes under
`data/permutation-10m-v3`; it does not overwrite or mutate
`data/permutation-10m-v2`.  Generation took 43.76 seconds, and full
record-by-record mathematical and encoding verification took 34.59 seconds.

Parent manifest SHA-256:

```text
b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f
```

The checked-in [public parent-manifest
snapshot](manifests/permutation-10m-v3.json) is byte-identical to the local
source manifest.  The full verifier run is recorded in the
[v3 verification report](manifests/permutation-10m-v3-verification.json).

The completed split is shard-based:

| Split | Shards | Records | Manifest SHA-256 |
|---|---|---:|---|
| Train | `000-097` | 9,800,000 | `7ad40c63a7559c52640d233a5398125d14160d83acadfa30637de291292893fa` |
| Validation | `098` | 100,000 | `90e88845f3f58947f317c67144c83bc5e38c27b248227e311632af834d2fd068` |
| Test | `099` | 100,000 | `3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b` |

Dataset readiness is established by the manifest and successful full verifier.
Model readiness and completion are separate; no v3 model has been trained yet.

## Shared corpus unit and sampling

One data record is one causal-language-model sequence containing one task. The
production corpus for either version contains 10,000,000 records and is exactly
balanced across that version's 20 tasks. Permutation size is sampled uniformly from 2 through 30,
unless a task requires a deliberately balanced construction. Sampling is
deterministic from the global seed and shard number; duplicates are allowed.

All permutations belong to the symmetric group `S_n` and use 1-based values
and positions. Numbers use canonical big-endian base 100. Values 0 through 99
are single two-digit tokens. Values of 100 or more are wrapped in
`<NUM_START>` and `<NUM_END>`.

## The v2 baseline tasks

The list below is frozen historical documentation for `permutation-20/v2`.
For v3, replace items 16--18 according to the revision table above.

### Encoding / translation

1. `to_cycle`: canonical disjoint cycles. Singleton cycles are included; each
   cycle begins at its least value; cycles are ordered by their least values.
2. `to_lehmer`: `L_i = #{j > i : pi_j < pi_i}`.
3. `to_inversion_vector`: value-indexed
   `I_v = #{u > v : position(u) < position(v)}`.
4. `to_reduced_word`: the deterministic reduced word in adjacent generators
   returned by stable bubble sorting. Products act on the right, so `pi s_i`
   swaps positions `i` and `i+1`.

### Statistics / properties

5. `length`: Coxeter length, equal to inversion count.
6. `descents`: the number of indices `i` with `pi_i > pi_(i+1)`.
7. `fixed_points`: the number of `i` with `pi_i = i`.
8. `parity`: inversion count modulo 2 (`00` even, `01` odd).
9. `cycle_type`: all cycle lengths, including 1s, sorted in decreasing order.
10. `rsk_shape`: row lengths of the RSK insertion tableau, listed in decreasing
    order.
11. `lis_length`: longest strictly increasing subsequence length.
12. `lds_length`: longest strictly decreasing subsequence length.
13. `pattern_avoidance`: whether `pi` avoids the supplied classical pattern;
    `01` means avoids and `00` means contains.

### Algebraic operations / comparisons

Function composition is `(a o b)(i) = a(b(i))`.

14. `inverse`: `pi^-1`.
15. `compose`: `pi o sigma` for the supplied second permutation `sigma`.
16. `power`: `pi^k` for a supplied nonnegative `0 <= k <= 100`.
17. `conjugate`: `g o pi o g^-1` for the supplied conjugator `g`.
18. `commutator`: `[pi,sigma] = pi o sigma o pi^-1 o sigma^-1`.
19. `right_multiply_simple`: `pi s_i`, swapping one-line positions `i` and
    `i+1`, for `1 <= i < n`.
20. `bruhat_leq`: strong Bruhat comparison `u <= v`. Positive and negative
    examples are matched by permutation size and positive Coxeter-length gap
    (gaps 1 through 4), so the label cannot be inferred from inversion counts.
    The positives include non-cover comparable pairs and the negatives are
    incomparable pairs. With
    `r_w(p,q) = #{i <= p : w(i) <= q}`, this holds exactly when
    `r_u(p,q) >= r_v(p,q)` for every `p,q`.

## Passage Math grammar

The supplied prefix and scalar-task layout is preserved:

```text
<BOS> <SIZE> ENCODE(n) PRIMARY [OPERANDS] <TASK> = ANSWER <EOS>
```

`PRIMARY` is one-line notation bounded by `<ONE_START>` and `<ONE_END>`.
Entries are comma separated. A second permutation uses `<ARG_START>` and
`<ARG_END>`. Pattern, exponent, and simple-reflection operands have their own
typed boundary/label tokens. Structured answers have typed boundaries; scalar
and Boolean answers use number encoding. There is exactly one task token and
one answer in every sequence, so the end-of-input, task, and equals positions
remain useful mechanistic-interpretability landmarks.

Example inherited from the supplied format:

```text
<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END> <DESCENTS> = 02 <EOS>
```

The authoritative token construction is implemented in
`neurips_permutations.passage`; tests freeze every operand and answer form.

## Reproducibility and storage

- JSON Lines is used as the container; each line stores an ID, task, size,
  token list, canonical space-separated text, and minimal structured metadata.
- Production shards use deterministic gzip with no timestamp in the header.
- Files are written to a temporary name, flushed and fsynced, then atomically
  renamed.
- A manifest contains SHA-256 hashes and exact per-task counts.
- Data shards are not committed to Git because GitHub is not suitable for the
  resulting multi-gigabyte artifact.

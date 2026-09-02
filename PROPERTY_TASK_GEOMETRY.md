# Combinatorial Task-Geometry Study

## Primary question

> At a fixed task count, does mathematical correspondence between two task
> sets predict the similarity of the representations learned by causal
> Transformers?

The study asks whether neural task geometry recovers human-known
combinatorial structure. It complements work that fixes a task and varies the
input representation, and it controls a confound in studies where task count
and the probability of including related tasks increase together.

The protocol is frozen in
[`configs/property_task_geometry.toml`](configs/property_task_geometry.toml).
No test record is used for selection, training diagnostics, CKA, or model
selection.

## Experiment 1: single-task geometry

Sixteen specialists are trained independently for each of three model seeds.
They cover eight preregistered relations:

| Left task | Right task | Input transformation | Exact label relation |
|---|---|---|---|
| Descents | Recoils | inverse | equal |
| Peaks | Valleys | value complement | equal |
| Fixed points | Anti-fixed points | value complement | equal |
| Exceedances | Deficiencies | inverse | equal |
| LIS length | LDS length | value complement | equal |
| Longest increasing run | Longest decreasing run | value complement | equal |
| Left-to-right maxima | Left-to-right minima | value complement | equal |
| Global descents | Components | value complement | right equals left plus one |

All eight identities were exhaustively verified through every permutation in
`S_2, ..., S_8`. The primary specialist contrast is:

```text
same task across seeds  vs  direct-relation tasks  vs  no-direct-relation tasks
```

Seed-level CKA values are first aggregated within a task pair. Task pairs, not
individual seed combinations, are the mathematical units of evidence. The
analysis also correlates the preregistered relation matrix with the learned
task-by-task CKA matrix using task-label permutations rather than treating
dependent matrix entries as independent observations.

## Experiment 2: composition at fixed task count

Every model in this experiment learns exactly four tasks. For each bundle
layout, one anchor model is reused in four comparisons. The corresponding B
model contains exactly `r = 0, 1, 2, or 4` direct counterparts of the anchor
tasks, while sharing no task name with the anchor.

The design crosses four independently selected bundle layouts with seeds
`17`, `42`, and `101`, giving 12 paired CKA curves. Across the four layouts:

- the anchor bundles contain every selected task exactly once;
- the B bundles at each value of `r` contain every selected task exactly once;
- architecture, parameter count, context, update count, effective batch,
  task exposure, data split, encoding, and probe inputs are fixed;
- direct correspondence count is exactly `0`, `1`, `2`, or `4`.

This balance prevents a condition mean from being driven by one task appearing
more frequently. Non-dual label correlations are measured after separately
standardizing each property within every permutation length and are retained
as an explicit diagnostic covariate.

The layout search was frozen before training. It sampled 60,000 deterministic
anchor candidates, retained 300 by the `r=4` non-dual-correlation score, then
jointly balanced the `r=0` and `r=4` scores. For the chosen anchors, the
balanced `r=1` and `r=2` layouts and task orientations were exhaustively
enumerated. The exact objectives and search seed are in the TOML file.

The primary endpoint is the paired final-layer difference
`CKA(r=4) - CKA(r=0)` across the 12 layout-by-seed cells. The report also
includes all four condition means, all individual curves, layerwise results,
Spearman trends, monotonicity counts, a paired sign/permutation analysis, and
split/seed variance decomposition. A larger CKA value is not assumed to be
intrinsically better: related tasks could instead produce a larger, structured
change in the anchor representation.

## Experiment 3: symmetry mechanism

The same specialist checkpoints are evaluated on three versions of an
identical 4,096-example validation probe:

```text
identity permutation
inverse permutation
value-complemented permutation
```

For every direct pair, the right-hand model is compared under the mathematically
correct transformation, the identity transformation, and the other
transformation as a negative control. For example:

```text
CKA(H_descents(pi), H_recoils(inverse(pi)))
CKA(H_LIS(pi), H_LDS(complement(pi)))
```

This distinguishes generic similarity from alignment that is specifically
consistent with the known combinatorial symmetry.

## Models, data, and accounting

- standard four-layer pre-LN causal Transformer;
- `d_model=256`, eight heads, FFN width 1,024, 3,240,448 parameters;
- one-line Passage Math input, 128-token context, tied embeddings;
- 20,000 AdamW updates, bf16, effective batch size 64;
- 16M-record scalar-property corpus with 500,000 records per property;
- 15.68M/160k/160k train/validation/test split.

Every 1,000 updates, validation evaluates the same first 160 records for each
of the 32 properties. The implementation collects those task-specific
prefixes in one physical shard scan and then retains the original independent
token-budget batches, bf16 forwards, and metric accumulation. On a completed
legacy checkpoint, all 160 scalar fields across 32 tasks and five metrics
matched the historical one-scan-per-task path exactly (`max_abs_diff = 0`),
while full validation took 1.41 seconds. This is an I/O optimization only; it
does not change training examples, optimization, validation examples, or CKA
probes.

There are 48 specialist positions and 60 four-task positions, totaling 108
unique model designs. Exact task/seed checkpoints from earlier protocols are
reused only after their data, architecture, optimizer schedule, task, seed,
step count, and checkpoint hash are verified. They are never counted as new
replicates or trained again.

## Results

All 108 model positions completed 20,000 updates and passed checkpoint,
configuration, finite-tensor, validation-grid, and provenance checks. The
analysis uses the same 4,096 task-free validation prefixes throughout and does
not read the test split. Full tables and provenance are in the
[result package](results/property-task-geometry/cka/README.md).

For the single-task specialists, final-layer CKA is `0.7096 +/- 0.2139` for
the same task across seeds, `0.1375 +/- 0.1454` for the eight preregistered
direct relations, and `0.0903 +/- 0.0571` for the other 112 cross-task pairs.
The direct-minus-other contrast is positive under a 100,000-sample task-label
permutation test (`p=0.015`).

The symmetry mechanism supplies the clearest evidence. Applying the
mathematically correct inverse or complement increases CKA over identity by
`0.3832 +/- 0.2510` and over the wrong transformation by
`0.4152 +/- 0.2329`. Both paired contrasts are positive in all 24 pair-seed
units. Because seeds are clustered within mathematical relations, the
pair-seed sign-test value (`p=1.19e-7`) is descriptive rather than the primary
inference. After averaging over seeds, both contrasts remain positive in all
eight relations (two-sided relation-level exact sign-test `p=0.0078125` for
each).

The fixed-four-task composition experiment is negative for a monotonic dose
response. Mean final-layer CKA at `r=0,1,2,4` direct correspondences is
`0.2935`, `0.2752`, `0.2662`, and `0.3648`; only one of 12 paired curves is
monotonic. The paired `r=4-r=0` change is `+0.0712 +/- 0.2121`, is positive in
7/12 cells, and has two-sided exact sign-test `p=0.774`. Thus, these results
support transformation-specific neural alignment but do not support the
broader claim that increasing related-task content reliably increases CKA.

## Interpretation boundary

The labels `direct relation` and `no direct relation` refer to the frozen eight
relations above. They do not claim that other statistic pairs are
mathematically independent. Shared input syntax, label correlation, task
difficulty, initialization, and optimization can all affect CKA; therefore the
paper reports same-task ceilings, random-initialization controls, behavioral
accuracy, conditional label correlations, and every unaveraged pair result.

The result can support the claim that neural geometry reflects a specified
hierarchy of combinatorial relationships. It cannot by itself establish a
shared algorithm, causal transfer, or a universal permutation representation.

## Related-work distinction

[Scullen, Kvinge, and Jenne (2026)](https://proceedings.mlr.press/v334/scullen26a.html)
fix the mathematical task and vary the permutation input representation. This
study fixes one-line representation and varies the mathematical task.

Recent work on
[convergent world representations](https://arxiv.org/abs/2602.00533) reports
that increasing task count can increase CKA even for disjoint task sets. This
protocol instead fixes the task count at four and changes the number of
preregistered mathematical correspondences.

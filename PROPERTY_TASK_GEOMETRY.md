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

There are 48 specialist positions and 60 four-task positions, totaling 108
unique model designs. Exact task/seed checkpoints from earlier protocols are
reused only after their data, architecture, optimizer schedule, task, seed,
step count, and checkpoint hash are verified. They are never counted as new
replicates or trained again.

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

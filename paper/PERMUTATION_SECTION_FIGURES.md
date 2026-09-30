# Permutation-section figures

This directory maps the five figures requested by the current permutation
draft to their result files, statistical units, and paper-ready captions. Both
SVG and 300-dpi PNG outputs are generated from checked CSV inputs; no plotted
value is typed into the plotting code.

## Figure 1: hard zero-shot execution

Files:

- [`permutation_fig1_zero_shot.svg`](figures/permutation-section/permutation_fig1_zero_shot.svg)
- [`permutation_fig1_zero_shot.png`](figures/permutation-section/permutation_fig1_zero_shot.png)

Suggested caption:

> **Hard zero-shot execution across training-property counts.** Exact accuracy
> on the opposite, disjoint property pool increases from
> $12.25\pm2.66\%$ at $k=1$ to $16.72\pm2.50\%$ at $k=16$, but the trend is
> non-monotonic and remains below the $32.83\%$ task-specific majority
> baseline. Gray lines show the three joint property-partition/model-seed
> replicates; blue points and error bars show their mean and sample standard
> deviation.

## Figure 2: final-layer linear probing

Files:

- [`permutation_fig2_linear_probe.svg`](figures/permutation-section/permutation_fig2_linear_probe.svg)
- [`permutation_fig2_linear_probe.png`](figures/permutation-section/permutation_fig2_linear_probe.png)

Suggested caption:

> **Unseen-property information becomes more linearly decodable with broader
> multitask training.** Final-layer, length-conditioned probe $R^2$ is 0.207,
> 0.259, 0.272, 0.308, and 0.301 for $k=1,2,4,8,16$, respectively. The dashed
> line is the matched random-initialization mean of 0.215. Error bars are sample
> standard deviations over three joint property-partition/model-seed
> replicates. The base Transformers are frozen, and probe performance does not
> imply that the model can execute an unseen task.

## Figure 3: layer-wise linear probing

Files:

- [`permutation_fig3_layerwise_probe.svg`](figures/permutation-section/permutation_fig3_layerwise_probe.svg)
- [`permutation_fig3_layerwise_probe.png`](figures/permutation-section/permutation_fig3_layerwise_probe.png)

Suggested caption:

> **Property information becomes more linearly accessible across Transformer
> depth.** For the $k=16$ models, prediction error (one minus rounded exact
> probe accuracy) falls from 61.28% at the embedding layer to 48.32% for
> trained properties and from 61.26% to 50.45% for opposite-pool properties at
> Block 4. Final normalization causes a small increase. Curves and error bars
> show the mean and sample standard deviation across the three joint
> replicates; the dashed line is the length-mode error baseline.

This figure and Figure 2 use the independent four-layer probe retraining
snapshot. The 50.45% unseen Block 4 value therefore supersedes the 50.38%
value from the earlier checkpoint set in the draft text.

## Figure 4: task relations and symmetry-aware alignment

Files:

- [`permutation_fig4_task_relations.svg`](figures/permutation-section/permutation_fig4_task_relations.svg)
- [`permutation_fig4_task_relations.png`](figures/permutation-section/permutation_fig4_task_relations.png)

Suggested caption:

> **Representation alignment reflects specified combinatorial relations.**
> (a) Among 48 single-property specialists, the eight directly related task
> pairs have higher final-layer CKA than the other 112 pairs (0.1375 versus
> 0.0903; task-label permutation $p=0.015$). Bars show means with sample
> standard deviations and dots show task-pair values. (b) Aligning model inputs
> using the mathematically correct inverse or complement transformation raises
> CKA above identity and wrong-transform controls for all eight preregistered
> relations (relation-level two-sided sign test $p=0.0078$). CKA measures
> representation alignment rather than behavioral accuracy.

## Figure 5: number of related pairs

Files:

- [`permutation_fig5_related_pair_count.svg`](figures/permutation-section/permutation_fig5_related_pair_count.svg)
- [`permutation_fig5_related_pair_count.png`](figures/permutation-section/permutation_fig5_related_pair_count.png)

Suggested caption:

> **Increasing the number of direct relations does not produce a monotonic CKA
> law.** With four training properties per model, mean final-layer CKA is
> 0.2935, 0.2752, 0.2662, and 0.3648 for 0, 1, 2, and 4 direct relations,
> respectively. Gray lines show 12 paired bundle-layout/seed cells; blue points
> and error bars show their mean and sample standard deviation. Only one of 12
> paired curves is monotonic, and the paired $r=4$ versus $r=0$ sign test gives
> $p=0.774$.

## Reproduction and provenance

Generate all ten image files and the SHA-256 manifest with:

```bash
pip install -e '.[figures]'
permutation-manuscript-figures \
  --repository . \
  --output-dir paper/figures/permutation-section
```

[`manifest.json`](figures/permutation-section/manifest.json) records the exact
input and output hashes. Figures 1, 4, and 5 use the checked-in Property32
behavior and task-geometry summaries. Figures 2 and 3 use a compact frozen
snapshot of the independent four-layer retraining from PermuFormer2 commit
`b6a9d6eac4ec94263b29edba4611d70277c2bfb0`. CKA is computed on validation
prefixes; it should not be described as test-set accuracy.

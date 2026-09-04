# Four-page paper figure set

The permutation contribution is compressed into two main-text composite
figures and one supplementary diagnostic. Do not add separate plots for each
experiment: they would repeat the same task-count axis and leave too little
space for the integer study, methods, and limitations.

## Recommended four-page layout

1. Use **Figure 1** in the main text. It is the compact summary of Henry
   Kvinge's three generalization questions: hard zero-shot behavior, linear
   probing, and twenty-shot adaptation.
2. Use **Figure 2** in the main text if task geometry is the paper's main
   representation result. If the integer section needs a full-width figure,
   move Figure 2 to the supplement and retain its result in one sentence.
3. Keep **Figure S1** in the supplement. The main-text scaling result needs
   only one sentence: every structured-holdout exact accuracy is zero, even
   after tenfold exposure and doubled depth.

Both main figures are 1,200 pixels wide in SVG coordinates and 2,400 pixels
wide in the 300-dpi PNG export. They are designed for a full-width `figure*`
in a two-column layout. Use the SVG for editing and the PNG for direct
Overleaf inclusion.

## Figure 1: three generalization signals

Files:

- [`figure1_generalization_signals.svg`](figures/figure1_generalization_signals.svg)
- [`figure1_generalization_signals.png`](figures/figure1_generalization_signals.png)

Recommended caption:

> **Three forms of permutation generalization under zero-overlap task
> training.** (a) Opposite-pool exact accuracy increases modestly with the
> number of base-training tasks but remains far below the task-specific
> majority baseline. (b) Final-layer length-conditioned probe $R^2$ increases
> through $k=8$ and then declines slightly at $k=16$; the dashed line is the
> random-initialization control. (c) The twenty-shot pretrained-minus-random
> exact-accuracy contrast grows progressively at learning rate $10^{-5}$ but
> is smaller and non-monotonic at $3\times10^{-4}$. Error bars are sample
> standard deviations over three joint task-split/model-seed replicates.

Main-text takeaway:

> Broader multitask training produces progressively stronger internal and
> low-rate adaptation signals, but it does not yield reliable hard zero-shot
> execution, and the few-shot advantage depends on optimization.

## Figure 2: task geometry and mathematical correspondence

Files:

- [`figure2_task_geometry.svg`](figures/figure2_task_geometry.svg)
- [`figure2_task_geometry.png`](figures/figure2_task_geometry.png)

Recommended caption:

> **Learned representation geometry reflects specific combinatorial
> correspondences rather than task count alone.** (a) Final-layer CKA between
> disjoint task pools rises sharply at $k=8$ but is non-monotonic across three
> independently frozen task-split/model-seed replicates. (b) Single-task
> specialists for directly related properties have higher CKA than other
> cross-task pairs (task-label permutation $p=0.015$), although both remain
> far below same-task alignment. (c) Relating probe inputs by the correct
> inverse or complement symmetry raises CKA above identity and wrong-transform
> controls in all eight preregistered mathematical relations (relation-level
> two-sided sign test $p=0.0078$). CKA measures representation alignment, not
> behavioral accuracy.

Main-text takeaway:

> Representation similarity is most convincing when the comparison encodes a
> known mathematical transformation; simply adding more or more-related tasks
> does not produce a monotonic CKA law.

## Figure S1: scaling diagnostics

Files:

- [`figureS1_scaling_diagnostics.svg`](figures/figureS1_scaling_diagnostics.svg)
- [`figureS1_scaling_diagnostics.png`](figures/figureS1_scaling_diagnostics.png)

Recommended caption:

> **Secondary diagnostics for the preregistered $k=16$ scaling factorial.**
> Tenfold training exposure, doubled depth, and their combination leave exact
> accuracy at zero on reduced-word translation, composition, and Lehmer-code
> translation in all 24 architecture-condition-seed endpoints. Teacher-forced
> token accuracy and answer-token loss move in different directions across
> Transformer and MLP models, so they do not provide evidence of a general
> scaling improvement. Bars show means and sample-standard-deviation error
> bars over three paired seeds.

## LaTeX snippets

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/figure1_generalization_signals.png}
  \caption{Three forms of permutation generalization under zero-overlap task training.}
  \label{fig:permutation-generalization}
\end{figure*}
```

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/figure2_task_geometry.png}
  \caption{Learned representation geometry reflects specific combinatorial correspondences.}
  \label{fig:permutation-geometry}
\end{figure*}
```

Use the full captions above in the submitted paper; the snippets keep the
example short. Figure S1 should use the same pattern in the supplement.

## Reproduction and provenance

Regenerate all six image files and their SHA-256 manifest with:

```bash
pip install -e '.[figures]'
permutation-paper-figures --repository . --output-dir paper/figures
```

[`figures/manifest.json`](figures/manifest.json) records every source CSV and
output-image hash. No value is entered manually in the plotting code.

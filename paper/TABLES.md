# Final paper tables

The four-page paper should use only Table 1 below in the main text. It aligns
Henry Kvinge's three generalization questions with the fixed-seed CKA control
without mixing incompatible metrics into one average. The auxiliary extension
table is supplementary material.

## Table 1: task-count trends

| Outcome | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---:|---:|---:|---:|---:|
| Hard zero-shot exact accuracy (%) | 12.25 +/- 2.66 | 11.49 +/- 2.22 | 11.96 +/- 1.90 | 13.64 +/- 3.73 | 16.72 +/- 2.50 |
| Final-layer linear-probe R2 | 0.198 +/- 0.077 | 0.245 +/- 0.034 | 0.271 +/- 0.015 | 0.307 +/- 0.017 | 0.297 +/- 0.014 |
| 20-shot pretrained minus random exact accuracy (pp), LR=1e-5 | -5.30 +/- 7.71 | -1.37 +/- 9.66 | +3.96 +/- 8.42 | +9.82 +/- 4.50 | +11.71 +/- 5.24 |
| Fixed-seed disjoint-pool final-layer CKA | 0.101 +/- 0.112 | 0.175 +/- 0.135 | 0.302 +/- 0.160 | 0.644 +/- 0.154 | 0.600 +/- 0.180 |

Values are mean +/- sample SD over three replicates. The first three rows use
the original R0/R1/R2 joint task-split/model-seed replicates. The CKA row uses
R0/R3/R4 with Transformer seed fixed at 17, so its variation is due to the
balanced task partition. Hard zero-shot accuracy remains below the 32.83%
task-macro majority baseline at every k. Probe R2 measures linear decodability,
not direct task execution. The fine-tuning contrast is validation-only because
it is the matched-learning-rate sensitivity analysis.

Recommended caption:

> **Generalization changes with base-training task count, but the signals
> differ by evaluation regime.** Direct zero-shot execution remains below a
> majority baseline. Linear decodability and low-rate few-shot transfer improve
> through broader training, while fixed-seed cross-pool CKA increases strongly
> but is not strictly monotonic. Error bars are sample standard deviations over
> three replicates; see the text for the two replicate definitions.

## Table S1: optional extension outcomes

| Extension | Primary result | Interpretation |
|---|---|---|
| Paired 5/20/100-shot curve | No consistent support-size improvement; Transformer k=4 exact accuracy rises from 8.65% to 11.20%, while several other cells are flat or decline. | More unique support data alone does not explain the few-shot trend. |
| n=31--40 extrapolation | Exact accuracy collapses for both architectures and every k relative to n=2--30. | The learned procedures do not reliably extrapolate beyond the training-length range. |
| 10x data / 2x depth scaling | Structured-holdout exact accuracy is 0% in all 24 endpoints. | Exposure and depth do not rescue opaque unseen-operation execution. |
| Four-representation transfer | Held-out exact accuracy is 30.61 +/- 2.64%; exact minus majority is +11.46 +/- 2.64 pp. | Grounding task tokens and representations enables partial but cell-dependent transfer. |

## LaTeX for Table 1

```latex
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
Outcome & $k=1$ & $k=2$ & $k=4$ & $k=8$ & $k=16$ \\
\midrule
Zero-shot exact (\%) & $12.25{\pm}2.66$ & $11.49{\pm}2.22$ & $11.96{\pm}1.90$ & $13.64{\pm}3.73$ & $16.72{\pm}2.50$ \\
Probe $R^2$ & $.198{\pm}.077$ & $.245{\pm}.034$ & $.271{\pm}.015$ & $.307{\pm}.017$ & $.297{\pm}.014$ \\
20-shot pretrained--random (pp) & $-5.30{\pm}7.71$ & $-1.37{\pm}9.66$ & $+3.96{\pm}8.42$ & $+9.82{\pm}4.50$ & $+11.71{\pm}5.24$ \\
Fixed-seed cross-pool CKA & $.101{\pm}.112$ & $.175{\pm}.135$ & $.302{\pm}.160$ & $.644{\pm}.154$ & $.600{\pm}.180$ \\
\bottomrule
\end{tabular}
\caption{Generalization and representation trends as the number of base-training tasks increases.}
\label{tab:permutation-main}
\end{table*}
```

All numbers are copied from checked-in CSVs; none are computed from rounded
values. The source files are listed in [FIGURES.md](FIGURES.md) and the result
index at [results/README.md](../results/README.md).

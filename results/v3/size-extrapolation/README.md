# Permutation-length extrapolation (n=31-40)

Thirty completed v3 nested models were evaluated on 1,000 new examples 
per task at lengths 31 through 40. The reduced-word task is excluded 
because some outputs exceed the frozen 1,024-token context. Values below 
are task-macro exact-sequence accuracy, averaged within seed and then 
reported as mean plus/minus sample standard deviation across three seeds.

| Architecture | k | Status | n=2-30 | n=31-40 | Change |
|---|---:|---|---:|---:|---:|
| MLP | 1 | fixed_train_holdout | 2.28% | 0.97% +/- 0.73% | -1.32 pp |
| MLP | 1 | pool_unseen | 8.38% | 0.98% +/- 0.30% | -7.40 pp |
| MLP | 1 | seen | 50.29% | 0.17% +/- 0.21% | -50.12 pp |
| MLP | 2 | fixed_train_holdout | 1.85% | 0.07% +/- 0.06% | -1.78 pp |
| MLP | 2 | pool_unseen | 6.83% | 0.39% +/- 0.14% | -6.44 pp |
| MLP | 2 | seen | 50.54% | 1.77% +/- 0.91% | -48.77 pp |
| MLP | 4 | fixed_train_holdout | 3.77% | 4.26% +/- 1.84% | +0.49 pp |
| MLP | 4 | pool_unseen | 4.25% | 0.80% +/- 0.48% | -3.45 pp |
| MLP | 4 | seen | 44.14% | 2.52% +/- 1.00% | -41.62 pp |
| MLP | 8 | fixed_train_holdout | 0.00% | 5.13% +/- 2.80% | +5.13 pp |
| MLP | 8 | pool_unseen | 1.82% | 1.80% +/- 1.43% | -0.01 pp |
| MLP | 8 | seen | 39.55% | 4.08% +/- 0.97% | -35.48 pp |
| MLP | 16 | fixed_train_holdout | 1.27% | 0.37% +/- 0.32% | -0.91 pp |
| MLP | 16 | seen | 34.40% | 2.63% +/- 0.47% | -31.77 pp |
| Transformer | 1 | fixed_train_holdout | 1.83% | 0.00% +/- 0.00% | -1.83 pp |
| Transformer | 1 | pool_unseen | 7.51% | 1.63% +/- 0.10% | -5.89 pp |
| Transformer | 1 | seen | 76.25% | 4.03% +/- 0.93% | -72.21 pp |
| Transformer | 2 | fixed_train_holdout | 2.18% | 0.00% +/- 0.00% | -2.18 pp |
| Transformer | 2 | pool_unseen | 7.30% | 1.67% +/- 0.18% | -5.63 pp |
| Transformer | 2 | seen | 89.75% | 11.43% +/- 2.87% | -78.31 pp |
| Transformer | 4 | fixed_train_holdout | 2.07% | 0.01% +/- 0.02% | -2.06 pp |
| Transformer | 4 | pool_unseen | 7.52% | 2.44% +/- 1.28% | -5.08 pp |
| Transformer | 4 | seen | 68.02% | 21.26% +/- 5.93% | -46.76 pp |
| Transformer | 8 | fixed_train_holdout | 1.07% | 0.12% +/- 0.11% | -0.95 pp |
| Transformer | 8 | pool_unseen | 2.77% | 2.40% +/- 2.05% | -0.37 pp |
| Transformer | 8 | seen | 59.60% | 14.11% +/- 1.06% | -45.50 pp |
| Transformer | 16 | fixed_train_holdout | 0.40% | 0.00% +/- 0.00% | -0.40 pp |
| Transformer | 16 | seen | 48.58% | 18.35% +/- 0.80% | -30.23 pp |

This is a post-hoc distribution-shift diagnostic, not a preregistered test. The shift changes both sequence length and the set of atomic number tokens appearing as permutation values.

- [Unaveraged results](model_task_results.csv)
- [Seed-aggregated summary](summary.csv)

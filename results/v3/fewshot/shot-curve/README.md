# Paired 5/20/100-shot adaptation curve

This post-hoc extension varies only the number of support examples. The
support sets are strictly nested (`5` is a subset of `20`, which is a
subset of `100`), and every endpoint uses the same 200 update steps,
learning rates, four v3 holdout tasks, architectures, base models, and
three model seeds. Each reported value is a task macro within seed, then
mean plus/minus sample standard deviation across seeds.

## Exact-sequence accuracy across all four holdouts

| Architecture | Base k | 5-shot | 20-shot | 100-shot |
|---|---:|---:|---:|---:|
| Transformer | 1 | 3.30% +/- 1.48% | 3.37% +/- 1.25% | 3.13% +/- 0.89% |
| Transformer | 2 | 5.67% +/- 0.39% | 7.04% +/- 0.53% | 6.84% +/- 1.96% |
| Transformer | 4 | 8.65% +/- 2.00% | 11.01% +/- 0.26% | 11.20% +/- 0.26% |
| Transformer | 8 | 11.89% +/- 0.66% | 12.68% +/- 0.32% | 12.59% +/- 0.36% |
| Transformer | 16 | 12.33% +/- 0.54% | 12.49% +/- 0.56% | 12.74% +/- 0.01% |
| Mlp | 1 | 1.76% +/- 0.04% | 1.72% +/- 0.01% | 1.71% +/- 0.00% |
| Mlp | 2 | 1.60% +/- 0.19% | 1.61% +/- 0.17% | 1.59% +/- 0.15% |
| Mlp | 4 | 7.90% +/- 1.56% | 9.38% +/- 1.44% | 9.38% +/- 1.73% |
| Mlp | 8 | 4.28% +/- 0.93% | 5.07% +/- 1.42% | 5.24% +/- 1.27% |
| Mlp | 16 | 4.35% +/- 2.23% | 5.50% +/- 4.08% | 5.34% +/- 3.94% |

## Structured-output exact accuracy

This table excludes the scalar parity task and averages reduced word,
composition, and Lehmer translation.

| Architecture | Base k | 5-shot | 20-shot | 100-shot |
|---|---:|---:|---:|---:|
| Transformer | 1 | 0.00% | 0.00% | 0.00% |
| Transformer | 2 | 0.00% | 0.00% | 0.00% |
| Transformer | 4 | 0.00% | 0.00% | 0.00% |
| Transformer | 8 | 0.21% | 0.11% | 0.11% |
| Transformer | 16 | 0.62% | 0.08% | 0.03% |
| Mlp | 1 | 0.00% | 0.00% | 0.00% |
| Mlp | 2 | 0.00% | 0.00% | 0.00% |
| Mlp | 4 | 0.00% | 0.00% | 0.00% |
| Mlp | 8 | 0.00% | 0.00% | 0.00% |
| Mlp | 16 | 0.00% | 0.00% | 0.00% |

## Interpretation

The primary question is whether increasing support from 5 to 100
examples improves complete-answer accuracy more for pretrained models
than for matched randomly initialized controls. A larger token accuracy
alone is insufficient because delimiters and copied tokens can dominate
that metric. The 20-shot test result was inspected before this extension,
so the full curve is a fixed post-hoc robustness analysis rather than a
new untouched confirmatory test.

Observed support-size result: increasing support from 5 to 100 examples did not produce a consistent pretrained-model improvement. The largest Transformer all-task exact gain was about 2.55 percentage points at base k=4; the other k values changed much less or decreased. Exact accuracy on the three structured outputs remained essentially zero. Matched random-initialization controls also improved with more unique examples, so the curve does not provide strong evidence that pretraining created a robust few-shot operation learner.

All endpoints use 800 training presentations (200 steps times batch size
4). Consequently, each support example is reused about 160, 40, or 8
times at 5, 20, or 100 shots. This controls update compute but intentionally
does not control repetitions per unique example.

## Artifacts

- [All unaveraged model-task endpoints](model_task_results.csv)
- [All-task endpoint summary](summary.csv)
- [Structured-output summary](structured_summary.csv)
- [Per-task summary](task_summary.csv)
- [Paired adaptation gains](adaptation_gains.csv)
- [Matched 100-shot minus 5-shot changes](endpoint_delta.csv)

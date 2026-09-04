# k=16 Scaling Study: Paper-Ready Section

## Methods

We tested whether weak zero-shot execution of structured held-out permutation operations was limited by training exposure or model depth. At the fully populated `k=16` endpoint, we crossed 1x versus 10x training exposure with 1x versus 2x depth. The 1x condition used 1.28 million examples, four Transformer layers or one causal-MLP block, and 20,000 optimizer updates. The 10x exposure condition used 12.8 million examples and 200,000 updates; the 2x-depth condition used eight Transformer layers or two causal-MLP blocks. All other optimizer, tokenizer, task-mixture, and sequence-length settings were fixed.

Each factorial cell contains seeds 17, 42, and 314159 for both architectures. The primary outcome is exact complete-answer accuracy, macro-averaged within each model over three structured tasks held out from training: reduced-word translation, composition, and Lehmer-code translation. Parity is reported separately as a short Boolean diagnostic. Every final model was evaluated on the same frozen v3 test split with 5,000 examples per task. We first formed a three-task macro within each seed and then report the mean and sample standard deviation over the three paired seeds.

## Results

| Architecture | Data | Depth | Structured loss | Token accuracy | Exact accuracy | Parity exact |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 1x | 1x | 6.2055 +/- 0.8726 | 21.921% +/- 4.778% | 0.000% +/- 0.000% | 1.193% +/- 1.243% |
| MLP | 1x | 1x | 7.8868 +/- 1.2551 | 26.918% +/- 0.678% | 0.000% +/- 0.000% | 3.820% +/- 0.985% |
| Transformer | 10x | 1x | 13.5609 +/- 1.3866 | 14.409% +/- 4.822% | 0.000% +/- 0.000% | 0.200% +/- 0.346% |
| MLP | 10x | 1x | 8.8903 +/- 0.9193 | 31.179% +/- 0.179% | 0.000% +/- 0.000% | 8.887% +/- 7.665% |
| Transformer | 1x | 2x | 6.6041 +/- 1.4398 | 22.592% +/- 8.984% | 0.000% +/- 0.000% | 1.147% +/- 1.542% |
| MLP | 1x | 2x | 8.1911 +/- 0.5207 | 26.237% +/- 0.987% | 0.000% +/- 0.000% | 5.093% +/- 2.182% |
| Transformer | 10x | 2x | 11.6767 +/- 1.1396 | 15.248% +/- 5.896% | 0.000% +/- 0.000% | 0.447% +/- 0.774% |
| MLP | 10x | 2x | 10.5254 +/- 0.9340 | 27.189% +/- 2.429% | 0.000% +/- 0.000% | 4.800% +/- 1.415% |

Seed-paired exact-accuracy effects are:

| Architecture | 10x data at 1x depth | 2x depth at 1x data | 10x data at 2x depth | 2x depth at 10x data | Interaction |
|---|---:|---:|---:|---:|---:|
| Transformer | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp |
| MLP | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp | +0.000 +/- 0.000 pp |

These contrasts are descriptive estimates from three paired seeds. Positive accuracy contrasts indicate improvement; negative loss contrasts indicate improvement. The complete loss and token-accuracy contrasts are retained in `factorial_effects.csv`.

## Limitations

1. The experiment has only three seeds per cell, so the sample standard deviations quantify observed seed variability but do not support precise population-level inference.
2. The study fixes `k=16`; it tests one difficult endpoint rather than a scaling law across task counts.
3. The 10x-data intervention also uses 10x more optimizer updates and approximately 10x more training examples. It is therefore an exposure-and-compute intervention, not a pure corpus-size intervention at matched compute.
4. The 2x-model intervention doubles depth, not every architectural dimension; its parameter multiplier is architecture dependent.
5. The primary macro contains three structured outputs with different lengths and difficulties. Parity is excluded from that macro because a Boolean answer is not directly comparable.
6. The experiment measures exact behavioral transfer to task tokens held out from gradient updates. It does not identify whether failure comes from ungrounded task-token semantics, missing algorithms, or decoding errors.
7. Teacher-forced token accuracy and loss can improve through formatting or local-token prediction even when complete-answer exact accuracy remains low.

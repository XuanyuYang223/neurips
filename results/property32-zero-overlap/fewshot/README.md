# Property32 twenty-shot fine-tuning results

This is Henry Kvinge's fine-tuning notion of generalization on the same
30 zero-overlap Transformers used for CKA and linear probing. Each model
is adapted to four balanced opposite-pool properties using 20 support
examples. Final metrics use 2,500 examples per property from source shard
199, which was not used by the earlier linear-probe evaluation.

All 120 warm-start adaptations and 24 support-matched random-initialization
controls completed 200 updates and passed strict checkpoint audit.

## Primary result

| k | Loss | Token accuracy | Adapted exact | Zero-shot exact | Change from zero-shot | Change over random init |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.6717 +/- 0.9684 | 57.61% +/- 1.87% | 16.63% +/- 2.60% | 13.15% | +3.48 +/- 2.78 pp | -17.65 +/- 4.87 pp |
| 2 | 2.7179 +/- 0.5120 | 60.35% +/- 1.31% | 20.79% +/- 2.61% | 14.15% | +6.64 +/- 3.05 pp | -13.50 +/- 6.31 pp |
| 4 | 1.7988 +/- 0.2184 | 62.71% +/- 0.79% | 25.72% +/- 1.07% | 11.57% | +14.15 +/- 1.69 pp | -8.56 +/- 4.79 pp |
| 8 | 1.4693 +/- 0.4927 | 65.19% +/- 3.06% | 31.77% +/- 4.36% | 14.76% | +17.02 +/- 2.22 pp | -2.51 +/- 2.71 pp |
| 16 | 1.1416 +/- 0.2644 | 66.79% +/- 1.81% | 33.59% +/- 3.61% | 15.79% | +17.80 +/- 5.76 pp | -0.70 +/- 1.15 pp |

The random-initialization control reaches 34.28% +/- 4.38% exact accuracy.

Loss decreases and both accuracy measures improve as the base-training
set grows. Adapted exact accuracy rises from 16.63% at `k=1` to 33.59%
at `k=16`, and the paired improvement from zero-shot rises from +3.48 to
+17.80 percentage points. All three replicate endpoints move in the same
direction (R0 17.93% to 29.64%, R1 18.33% to 36.73%, R2 13.64% to 34.39%), although the intermediate points are not
strictly monotonic within every replicate.

The stronger conclusion does not hold: warm-start adaptation remains
below the random-initialization control at every `k`. Its mean deficit
shrinks from 17.65 points at `k=1` to 0.70 points at `k=16`, which is
smaller than the three-replicate error bar. Thus broader pretraining is
associated with greater 20-shot adaptability, but this experiment does
not establish a positive pretraining advantage over learning from scratch.

## Family heterogeneity at k=16

| Target family | Adapted exact | Change from zero-shot | Change over random init |
|---|---:|---:|---:|
| `local` | 27.39% +/- 5.54% | +11.04 +/- 2.57 pp | +1.23 +/- 1.55 pp |
| `positional` | 29.57% +/- 6.49% | +12.37 +/- 2.68 pp | +1.41 +/- 2.40 pp |
| `cycle` | 46.00% +/- 15.02% | +22.66 +/- 15.82 pp | -2.02 +/- 5.59 pp |
| `global_run` | 31.39% +/- 19.24% | +25.13 +/- 20.40 pp | -3.41 +/- 1.97 pp |

The target families differ substantially. At `k=16`, local and positional
properties have small positive mean contrasts over random initialization,
whereas cycle and global/run properties remain below it. These are
descriptive family macros over only three joint task-split/model-seed
replicates, not independent task-level significance tests.

## Aggregation and metric conventions

Means first average four target families and both pool directions
within each replicate, then report mean +/- sample SD across three joint
task-split/model-seed replicates. The primary contrast is improvement
from paired zero-shot accuracy; improvement over the seed- and
support-matched random control is the second confirmatory contrast.

`sequence_accuracy` requires the scalar answer and EOS to be correct and
is the primary complete-answer metric. `token_accuracy` scores those two
supervised tokens separately, so it is partly inflated by EOS. `loss` is
their mean negative log likelihood; lower is better. Every raw row has
2,500 examples and 5,000 supervised tokens.

Warm-start and random-init models use the same 20 support examples and 200
updates, but their frozen learning rates are `1e-5` and `3e-4`,
respectively. The random contrast therefore compares the two prespecified
adaptation recipes; it does not isolate initialization while holding the
learning rate fixed. Only three replicates are available, and task split
and model seed vary together.

This experiment measures few-shot adaptability, not hard zero-shot
execution. `model_task_results.csv` contains all 144 unaveraged rows;
`replicate_summary.csv`, `summary.csv`, and `family_summary.csv` expose
each aggregation stage. `random_summary.csv` contains the matched
random-initialization baseline.

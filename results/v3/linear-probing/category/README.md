# V3 category-model linear probing

Frozen ridge probes read task-free `<ONE_END>` activations from the 18
completed Encoding E4, Statistics S4, and Algebra A4 models. Every model
is evaluated on the same neutral battery of 32 scalar permutation
properties. Base-model weights remain frozen.

Probes are fitted and tuned on 8,192 validation
permutations and evaluated once on 8,192 independently
selected test permutations. Targets are standardized within permutation
length, so the primary R2 measures signal beyond a length-only baseline.

## Final-layer result

Values are property-macro means followed by sample SD across three seeds.
The delta is paired trained minus random initialization at the same
architecture and seed.

| Architecture | Training condition | R2 | R2 delta vs random | Exact | Exact delta vs random |
|---|---|---:|---:|---:|---:|
| Transformer | Encoding E4 | 0.3567 +/- 0.0128 | +0.1335 +/- 0.0152 | 47.98% +/- 0.54% | +2.67 +/- 0.54 pp |
| Transformer | Statistics S4 | 0.4314 +/- 0.0162 | +0.2083 +/- 0.0186 | 51.00% +/- 0.74% | +5.69 +/- 0.73 pp |
| Transformer | Algebra A4 | 0.2490 +/- 0.0034 | +0.0258 +/- 0.0047 | 45.19% +/- 0.27% | -0.12 +/- 0.26 pp |
| Transformer | Random init | 0.2232 +/- 0.0025 | -- | 45.31% +/- 0.04% | -- |
| MLP | Encoding E4 | 0.1592 +/- 0.0066 | +0.1563 +/- 0.0076 | 42.06% +/- 0.80% | +3.71 +/- 0.95 pp |
| MLP | Statistics S4 | 0.1306 +/- 0.0054 | +0.1278 +/- 0.0045 | 40.12% +/- 0.09% | +1.77 +/- 0.16 pp |
| MLP | Algebra A4 | 0.1322 +/- 0.0095 | +0.1293 +/- 0.0088 | 39.11% +/- 0.34% | +0.76 +/- 0.15 pp |
| MLP | Random init | 0.0029 +/- 0.0014 | -- | 38.35% +/- 0.22% | -- |

For the Transformer, Statistics S4 has the largest all-property R2
gain over random initialization, followed by Encoding E4; Algebra A4
adds little overall. For the MLP, all three trained conditions exceed
its weak random-feature baseline, with Encoding E4 highest overall.
The architecture-dependent ordering argues against a single universal
claim that one task family always produces the richest representation.

No exact scalar probe task was a category-model training target. Some
training outputs nevertheless determine probe labels: for example,
cycle type determines cycle-count properties and RSK shape determines
LIS and LDS lengths. The family table should therefore be interpreted
as mathematical alignment, not as 32 equally unrelated holdouts.

The complete CSVs separate local, positional, cycle, and global/run
target families at every layer. These are linear-decodability results,
not behavioral task accuracy and not evidence that the base model
causally uses a decoded property.

## Artifacts

- `model_property_layer_probes.csv`: every model/property/layer result;
- `model_family_macro_probes.csv`: property-family macros per model;
- `category_probe_summary.csv`: means and sample SD across seeds;
- `paired_random_contrasts.csv`: seed-paired trained-minus-random controls;
- `probe_manifests.json`, `run_provenance.json`, and `manifest.json`:
  sample, checkpoint, code, and artifact provenance.

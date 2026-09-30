"""Generate the five permutation figures referenced by the draft manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Sequence

from .paper_figures import (
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    LIGHT_GRAY,
    ORANGE,
    PALE_GRAY,
    PURPLE,
    SKY,
    TASK_COUNTS,
    PillowCanvas,
    SvgCanvas,
    _atomic_bytes,
    _axes,
    _error_bar,
    _legend_line,
    _map_y,
    _rows,
    _series,
)


LAYER_ORDER = ("embedding", "block_01", "block_02", "block_03", "block_04", "final_norm")
LAYER_LABELS = ("Embed", "Block 1", "Block 2", "Block 3", "Block 4", "Final norm")

MANUSCRIPT_FIGURE_FILES = (
    "permutation_fig1_zero_shot.svg",
    "permutation_fig1_zero_shot.png",
    "permutation_fig2_linear_probe.svg",
    "permutation_fig2_linear_probe.png",
    "permutation_fig3_layerwise_probe.svg",
    "permutation_fig3_layerwise_probe.png",
    "permutation_fig4_task_relations.svg",
    "permutation_fig4_task_relations.png",
    "permutation_fig5_related_pair_count.svg",
    "permutation_fig5_related_pair_count.png",
)

SOURCE_FILES = (
    "paper/figure_data/property32_l4_retrained_probe_summary.csv",
    "results/property32-zero-overlap/replicates/behavior_replicates.csv",
    "results/property32-zero-overlap/replicates/behavior_summary.csv",
    "results/property-task-geometry/cka/specialist_group_summary.csv",
    "results/property-task-geometry/cka/specialist_pair_summary.csv",
    "results/property-task-geometry/cka/symmetry_summary.csv",
    "results/property-task-geometry/cka/bundle_cell_cka.csv",
    "results/property-task-geometry/cka/bundle_summary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _title(canvas: Any, text: str, subtitle: str | None = None, *, width: int) -> None:
    canvas.text(width / 2, 30, text, size=16, bold=True, anchor="middle")
    if subtitle:
        canvas.text(width / 2, 51, subtitle, size=9.5, color=GRAY, anchor="middle")


def _figure1(canvas: Any, repository: Path) -> None:
    summaries = {
        int(row["trained_task_count"]): row
        for row in _rows(
            repository / "results/property32-zero-overlap/replicates/behavior_summary.csv"
        )
    }
    replicates = _rows(
        repository / "results/property32-zero-overlap/replicates/behavior_replicates.csv"
    )
    if set(summaries) != set(TASK_COUNTS):
        raise ValueError("Figure 1 requires the complete k grid")

    _title(
        canvas,
        "Hard zero-shot execution remains below the majority baseline",
        "Opposite-pool properties were absent from each model's base training.",
        width=760,
    )
    x, y, width, height = 92, 83, 600, 285
    positions = _axes(
        canvas,
        x=x,
        y=y,
        width=width,
        height=height,
        xlabels=tuple(str(k) for k in TASK_COUNTS),
        minimum=0.0,
        maximum=0.38,
        ticks=(0.0, 0.1, 0.2, 0.3),
        tick_format=lambda value: f"{100 * value:.0f}",
        metric_label="Exact answer accuracy (%)",
        x_padding=30,
    )
    replicate_ids = sorted({row["replicate_id"] for row in replicates})
    for replicate_id in replicate_ids:
        by_k = {
            int(row["trained_task_count"]): float(row["macro_sequence_accuracy"])
            for row in replicates
            if row["replicate_id"] == replicate_id
        }
        points = [_map_y(by_k[k], y, height, 0.0, 0.38) for k in TASK_COUNTS]
        for index in range(len(points) - 1):
            canvas.line(
                positions[index], points[index], positions[index + 1], points[index + 1],
                color=LIGHT_GRAY, width=1.25,
            )
        for px, py in zip(positions, points, strict=True):
            canvas.circle(px, py, 2.8, fill=LIGHT_GRAY, stroke=LIGHT_GRAY)

    means = [float(summaries[k]["macro_sequence_accuracy_mean"]) for k in TASK_COUNTS]
    sds = [float(summaries[k]["macro_sequence_accuracy_sample_sd"]) for k in TASK_COUNTS]
    _series(
        canvas, positions, means, sds, y=y, height=height,
        minimum=0.0, maximum=0.38, color=BLUE,
    )
    baseline = float(summaries[1]["macro_majority_baseline_sequence_accuracy_mean"])
    baseline_y = _map_y(baseline, y, height, 0.0, 0.38)
    canvas.line(x, baseline_y, x + width, baseline_y, color=ORANGE, width=1.8, dash="7 5")
    _legend_line(canvas, 348, 69, "mean +/- SD", BLUE)
    canvas.line(478, 65, 498, 65, color=LIGHT_GRAY, width=1.4)
    canvas.text(504, 69, "replicates", size=9, color=GRAY)
    canvas.line(590, 65, 610, 65, color=ORANGE, width=1.6, dash="7 5")
    canvas.text(616, 69, "majority baseline", size=9, color=GRAY)
    canvas.text(392, 411, "Number of training properties (k)", size=10, color=GRAY, anchor="middle")
    canvas.text(
        380, 445,
        "Means and sample SD use three joint task-partition/model-seed replicates.",
        size=9, color=GRAY, anchor="middle",
    )


def _retrained_probe_rows(repository: Path) -> list[dict[str, str]]:
    rows = _rows(repository / "paper/figure_data/property32_l4_retrained_probe_summary.csv")
    if len(rows) != 17:
        raise ValueError("the frozen L4 retraining snapshot must contain 17 rows")
    return rows


def _figure2(canvas: Any, repository: Path) -> None:
    rows = _retrained_probe_rows(repository)
    indexed = {
        (row["scope"], int(row["trained_task_count"]), row["layer"]): row for row in rows
    }
    _title(
        canvas,
        "Unseen properties become more linearly decodable",
        "Independent L4 retraining; the probe observes task-free <ONE_END> representations.",
        width=760,
    )
    x, y, width, height = 92, 83, 600, 285
    positions = _axes(
        canvas,
        x=x,
        y=y,
        width=width,
        height=height,
        xlabels=tuple(str(k) for k in TASK_COUNTS),
        minimum=0.15,
        maximum=0.36,
        ticks=(0.15, 0.20, 0.25, 0.30, 0.35),
        tick_format=lambda value: f"{value:.2f}",
        metric_label="Final-layer length-conditioned probe R2",
        x_padding=30,
    )
    selected = [indexed[("opposite_pool", k, "final_norm")] for k in TASK_COUNTS]
    means = [float(row["length_conditioned_r2_mean"]) for row in selected]
    sds = [float(row["length_conditioned_r2_sample_sd"]) for row in selected]
    _series(
        canvas, positions, means, sds, y=y, height=height,
        minimum=0.15, maximum=0.36, color=GREEN,
    )
    random_row = indexed[("random_init", 0, "final_norm")]
    random_value = float(random_row["length_conditioned_r2_mean"])
    random_y = _map_y(random_value, y, height, 0.15, 0.36)
    canvas.line(x, random_y, x + width, random_y, color=GRAY, width=1.8, dash="7 5")
    _legend_line(canvas, 400, 69, "trained models", GREEN)
    canvas.line(540, 65, 562, 65, color=GRAY, width=1.6, dash="7 5")
    canvas.text(568, 69, "random initialization", size=9, color=GRAY)
    canvas.text(392, 411, "Number of training properties (k)", size=10, color=GRAY, anchor="middle")
    canvas.text(
        380, 445,
        "Error bars are sample SD over three joint task-partition/model-seed replicates.",
        size=9, color=GRAY, anchor="middle",
    )


def _figure3(canvas: Any, repository: Path) -> None:
    rows = _retrained_probe_rows(repository)
    indexed = {
        (row["scope"], int(row["trained_task_count"]), row["layer"]): row for row in rows
    }
    _title(
        canvas,
        "Property information becomes linearly accessible across Transformer depth",
        "k=16; prediction error is one minus rounded exact probe accuracy.",
        width=760,
    )
    x, y, width, height = 92, 83, 600, 285
    positions = _axes(
        canvas,
        x=x,
        y=y,
        width=width,
        height=height,
        xlabels=LAYER_LABELS,
        minimum=0.45,
        maximum=0.64,
        ticks=(0.45, 0.50, 0.55, 0.60),
        tick_format=lambda value: f"{100 * value:.0f}",
        metric_label="Exact-property prediction error (%)",
        x_padding=20,
    )
    for scope, color in (("trained", BLUE), ("opposite_pool", ORANGE)):
        selected = [indexed[(scope, 16, layer)] for layer in LAYER_ORDER]
        means = [1.0 - float(row["exact_accuracy_mean"]) for row in selected]
        sds = [float(row["exact_accuracy_sample_sd"]) for row in selected]
        _series(
            canvas, positions, means, sds, y=y, height=height,
            minimum=0.45, maximum=0.64, color=color,
        )
    baseline = 1.0 - float(indexed[("trained", 16, "embedding")]["length_mode_baseline_accuracy_mean"])
    baseline_y = _map_y(baseline, y, height, 0.45, 0.64)
    canvas.line(x, baseline_y, x + width, baseline_y, color=GRAY, width=1.4, dash="6 4")
    _legend_line(canvas, 375, 69, "trained", BLUE)
    _legend_line(canvas, 463, 69, "unseen", ORANGE)
    canvas.line(550, 65, 570, 65, color=GRAY, width=1.4, dash="6 4")
    canvas.text(576, 69, "length baseline", size=9, color=GRAY)
    canvas.text(392, 411, "Representation depth", size=10, color=GRAY, anchor="middle")
    canvas.text(
        380, 445,
        "Lower is better; means and sample SD use three joint replicates.",
        size=9, color=GRAY, anchor="middle",
    )


def _figure4(canvas: Any, repository: Path) -> None:
    group_rows = {
        row["comparison"]: row
        for row in _rows(
            repository / "results/property-task-geometry/cka/specialist_group_summary.csv"
        )
    }
    pair_rows = _rows(
        repository / "results/property-task-geometry/cka/specialist_pair_summary.csv"
    )
    symmetry_rows = _rows(
        repository / "results/property-task-geometry/cka/symmetry_summary.csv"
    )
    symmetry: dict[str, dict[str, float]] = {}
    for row in symmetry_rows:
        symmetry.setdefault(row["pair_id"], {})[row["condition"]] = float(
            row["final_layer_cka_mean"]
        )
    if len(symmetry) != 8 or any(set(values) != {"identity", "wrong", "correct"} for values in symmetry.values()):
        raise ValueError("Figure 4 requires eight complete mathematical relations")

    _title(
        canvas,
        "Representation alignment reflects specified combinatorial relations",
        "Single-property specialists; final-layer linear CKA on 4,096 shared prefixes.",
        width=1200,
    )
    plot_y, plot_h, plot_w = 94, 280, 430

    canvas.text(32, 76, "(a) Related versus other task pairs", size=12.5, bold=True)
    x0 = 98
    positions = _axes(
        canvas,
        x=x0,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=("Direct relation", "Other pairs"),
        minimum=0.0,
        maximum=0.52,
        ticks=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        tick_format=lambda value: f"{value:.1f}",
        metric_label="Final-layer linear CKA",
        x_padding=105,
    )
    for px, key, color in zip(
        positions, ("direct_relation", "no_direct_relation"), (GREEN, SKY), strict=True
    ):
        values = [
            float(row["final_layer_cka_mean"])
            for row in pair_rows
            if row["comparison"] == key
        ]
        spread = 50 if len(values) > 10 else 34
        for index, value in enumerate(values):
            offset = ((index * 37) % 101) / 100 * 2 * spread - spread
            canvas.circle(
                px + offset, _map_y(value, plot_y, plot_h, 0.0, 0.52),
                2.0, fill=LIGHT_GRAY, stroke=LIGHT_GRAY,
            )
        mean = float(group_rows[key]["final_layer_cka_mean"])
        sd = float(group_rows[key]["final_layer_cka_sample_sd"])
        mean_y = _map_y(mean, plot_y, plot_h, 0.0, 0.52)
        canvas.line(px - 45, mean_y, px + 45, mean_y, color=color, width=4.0)
        _error_bar(
            canvas, px, mean, sd, y=plot_y, height=plot_h,
            minimum=0.0, maximum=0.52, color=BLACK,
        )
        canvas.text(px, mean_y - 11, f"{mean:.3f}", size=9, color=color, anchor="middle", bold=True)
    canvas.text(x0 + plot_w / 2, 434, "Task-pair class", size=10, color=GRAY, anchor="middle")
    canvas.text(x0 + plot_w / 2, 458, "Task-label permutation p = 0.015", size=9, color=GRAY, anchor="middle")

    canvas.text(626, 76, "(b) Input transformation reveals the relation", size=12.5, bold=True)
    x1 = 700
    conditions = ("identity", "wrong", "correct")
    positions = _axes(
        canvas,
        x=x1,
        y=plot_y,
        width=plot_w,
        height=plot_h,
        xlabels=("Identity", "Wrong control", "Correct transform"),
        minimum=0.0,
        maximum=1.0,
        ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
        tick_format=lambda value: f"{value:.2g}",
        metric_label="Relation-level final-layer CKA",
        x_padding=55,
    )
    for values in symmetry.values():
        mapped = [_map_y(values[condition], plot_y, plot_h, 0.0, 1.0) for condition in conditions]
        for index in range(2):
            canvas.line(
                positions[index], mapped[index], positions[index + 1], mapped[index + 1],
                color=LIGHT_GRAY, width=1.3,
            )
        for px, py in zip(positions, mapped, strict=True):
            canvas.circle(px, py, 2.7, fill=LIGHT_GRAY, stroke=LIGHT_GRAY)
    means = [statistics.fmean(values[c] for values in symmetry.values()) for c in conditions]
    mapped = [_map_y(value, plot_y, plot_h, 0.0, 1.0) for value in means]
    for index in range(2):
        canvas.line(
            positions[index], mapped[index], positions[index + 1], mapped[index + 1],
            color=PURPLE, width=3.4,
        )
    for px, py, mean in zip(positions, mapped, means, strict=True):
        canvas.circle(px, py, 4.4, fill=PURPLE)
        canvas.text(px, py - 11, f"{mean:.3f}", size=9, color=PURPLE, anchor="middle", bold=True)
    canvas.text(x1 + plot_w / 2, 434, "Alignment condition", size=10, color=GRAY, anchor="middle")
    canvas.text(
        x1 + plot_w / 2, 458,
        "Correct > both controls in 8/8 relations; sign-test p = 0.0078",
        size=9, color=GRAY, anchor="middle",
    )


def _figure5(canvas: Any, repository: Path) -> None:
    summary = {
        int(row["related_pair_count"]): row
        for row in _rows(repository / "results/property-task-geometry/cka/bundle_summary.csv")
    }
    raw = [
        row
        for row in _rows(repository / "results/property-task-geometry/cka/bundle_cell_cka.csv")
        if row["layer"] == "final_norm"
    ]
    related_counts = (0, 1, 2, 4)
    cell_ids = sorted({(row["split_id"], row["model_seed"]) for row in raw})
    if set(summary) != set(related_counts) or len(cell_ids) != 12:
        raise ValueError("Figure 5 requires four conditions and 12 paired cells")

    _title(
        canvas,
        "More direct task correspondences do not produce monotonic CKA",
        "Every model learns four properties; gray lines are paired bundle-layout/seed cells.",
        width=760,
    )
    x, y, width, height = 92, 83, 600, 285
    positions = _axes(
        canvas,
        x=x,
        y=y,
        width=width,
        height=height,
        xlabels=tuple(str(value) for value in related_counts),
        minimum=0.0,
        maximum=0.85,
        ticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        tick_format=lambda value: f"{value:.1f}",
        metric_label="Final-layer linear CKA",
        x_padding=42,
    )
    for split_id, seed in cell_ids:
        values = {
            int(row["related_pair_count"]): float(row["linear_cka"])
            for row in raw
            if row["split_id"] == split_id and row["model_seed"] == seed
        }
        mapped = [_map_y(values[count], y, height, 0.0, 0.85) for count in related_counts]
        for index in range(3):
            canvas.line(
                positions[index], mapped[index], positions[index + 1], mapped[index + 1],
                color=LIGHT_GRAY, width=1.1,
            )
        for px, py in zip(positions, mapped, strict=True):
            canvas.circle(px, py, 2.4, fill=LIGHT_GRAY, stroke=LIGHT_GRAY)
    means = [float(summary[count]["final_layer_cka_mean"]) for count in related_counts]
    sds = [float(summary[count]["final_layer_cka_sample_sd"]) for count in related_counts]
    _series(
        canvas, positions, means, sds, y=y, height=height,
        minimum=0.0, maximum=0.85, color=BLUE,
    )
    for px, mean in zip(positions, means, strict=True):
        py = _map_y(mean, y, height, 0.0, 0.85)
        canvas.text(px, py - 12, f"{mean:.3f}", size=9, color=BLUE, anchor="middle", bold=True)
    _legend_line(canvas, 400, 69, "mean +/- SD", BLUE)
    canvas.line(530, 65, 552, 65, color=LIGHT_GRAY, width=1.4)
    canvas.text(558, 69, "12 paired cells", size=9, color=GRAY)
    canvas.text(392, 411, "Number of direct relations across the two task sets", size=10, color=GRAY, anchor="middle")
    canvas.text(
        380, 445,
        "Only 1/12 curves is monotonic; paired r=4 minus r=0 sign-test p = 0.774.",
        size=9, color=GRAY, anchor="middle",
    )


def _render_pair(
    output_dir: Path,
    stem: str,
    width: int,
    height: int,
    renderer: Callable[[Any, Path], None],
    repository: Path,
) -> None:
    for suffix, canvas_type in (("svg", SvgCanvas), ("png", PillowCanvas)):
        canvas = canvas_type(width, height)
        renderer(canvas, repository)
        _atomic_bytes(output_dir / f"{stem}.{suffix}", canvas.finish())


def generate(repository: Path, output_dir: Path) -> dict[str, Any]:
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    missing = [name for name in SOURCE_FILES if not (repository / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing manuscript figure inputs: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    specifications = (
        ("permutation_fig1_zero_shot", 760, 470, _figure1),
        ("permutation_fig2_linear_probe", 760, 470, _figure2),
        ("permutation_fig3_layerwise_probe", 760, 470, _figure3),
        ("permutation_fig4_task_relations", 1200, 480, _figure4),
        ("permutation_fig5_related_pair_count", 760, 470, _figure5),
    )
    for stem, width, height, renderer in specifications:
        _render_pair(output_dir, stem, width, height, renderer, repository)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "status": "completed",
        "l4_retraining_source": {
            "repository": "https://github.com/uw-math-ai/PermuFormer2",
            "commit": "b6a9d6eac4ec94263b29edba4611d70277c2bfb0",
            "path": "camera-ready-2026-09/permutation/results/property32-zero-overlap/linear-probing-reproduction-full",
        },
        "inputs": {
            name: {"sha256": _sha256(repository / name), "bytes": (repository / name).stat().st_size}
            for name in SOURCE_FILES
        },
        "outputs": {
            name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in MANUSCRIPT_FIGURE_FILES
        },
    }
    _atomic_bytes(
        output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/permutation-section")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = generate(args.repository, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MANUSCRIPT_FIGURE_FILES", "generate", "main"]

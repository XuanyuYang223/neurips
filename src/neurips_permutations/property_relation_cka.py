"""CKA analysis for the frozen relation-controlled 3 x 3 study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .cka import (
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _git_commit,
    _load_trained_activations,
    _sha256,
    linear_cka,
    probe_identity,
    select_probe_examples,
)
from .property_relation_experiments import (
    DEFAULT_CONFIG,
    MODEL_SEEDS,
    SPLIT_IDS,
    TASK_COUNTS,
    RelationExperimentRun,
    build_relation_matrix,
    relation_summary,
)


DEFAULT_OUTPUT_DIR = Path("results/property32-relation-controlled/cka")
CELL_FIELDS = (
    "split_id",
    "model_seed",
    "trained_task_count",
    "final_layer_linear_cka",
)
SUMMARY_FIELDS = (
    "trained_task_count",
    "cell_count",
    "final_layer_linear_cka_mean",
    "final_layer_linear_cka_sample_sd",
    "final_layer_linear_cka_min",
    "final_layer_linear_cka_max",
)
LAYER_FIELDS = (
    "split_id",
    "model_seed",
    "trained_task_count",
    "layer",
    "probe_examples",
    "linear_cka",
    "run_id_a",
    "run_id_b",
)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def summarize_cell_curves(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_cells = {(split_id, seed) for split_id in SPLIT_IDS for seed in MODEL_SEEDS}
    observed_cells = {(str(row["split_id"]), int(row["model_seed"])) for row in rows}
    if observed_cells != expected_cells or len(rows) != 36:
        raise ValueError("relation-controlled CKA cell grid is incomplete")
    summaries: list[dict[str, Any]] = []
    for task_count in TASK_COUNTS:
        values = [
            float(row["final_layer_linear_cka"])
            for row in rows
            if int(row["trained_task_count"]) == task_count
        ]
        if len(values) != 9:
            raise ValueError(f"k={task_count} does not contain nine cells")
        summaries.append(
            {
                "trained_task_count": task_count,
                "cell_count": len(values),
                "final_layer_linear_cka_mean": statistics.fmean(values),
                "final_layer_linear_cka_sample_sd": statistics.stdev(values),
                "final_layer_linear_cka_min": min(values),
                "final_layer_linear_cka_max": max(values),
            }
        )

    cell_trends: dict[str, dict[str, Any]] = {}
    deltas: dict[tuple[str, int], float] = {}
    for split_id, seed in sorted(expected_cells):
        selected = sorted(
            (
                row
                for row in rows
                if row["split_id"] == split_id and int(row["model_seed"]) == seed
            ),
            key=lambda row: int(row["trained_task_count"]),
        )
        values = [float(row["final_layer_linear_cka"]) for row in selected]
        delta = values[-1] - values[0]
        deltas[split_id, seed] = delta
        cell_trends[f"{split_id}:{seed}"] = {
            "values": values,
            "delta_k8_minus_k1": delta,
            "spearman_rho": _pearson(list(range(4)), _ranks(values)),
            "monotonic_non_decreasing": all(
                left <= right for left, right in zip(values, values[1:])
            ),
            "peak_k": TASK_COUNTS[max(range(4), key=values.__getitem__)],
        }
    delta_values = list(deltas.values())
    positives = sum(value > 0 for value in delta_values)
    tail = sum(math.comb(9, i) for i in range(max(positives, 9 - positives), 10))
    sign_test_p = min(1.0, 2 * tail / (2**9))

    grand = statistics.fmean(delta_values)
    split_means = {
        split_id: statistics.fmean(deltas[split_id, seed] for seed in MODEL_SEEDS)
        for split_id in SPLIT_IDS
    }
    seed_means = {
        seed: statistics.fmean(deltas[split_id, seed] for split_id in SPLIT_IDS)
        for seed in MODEL_SEEDS
    }
    total_ss = sum((value - grand) ** 2 for value in delta_values)
    split_ss = len(MODEL_SEEDS) * sum((value - grand) ** 2 for value in split_means.values())
    seed_ss = len(SPLIT_IDS) * sum((value - grand) ** 2 for value in seed_means.values())
    return summaries, {
        "cell_trends": cell_trends,
        "delta_k8_minus_k1_mean": grand,
        "delta_k8_minus_k1_sample_sd": statistics.stdev(delta_values),
        "positive_delta_cells": positives,
        "two_sided_exact_sign_test_p": sign_test_p,
        "monotonic_cell_count": sum(
            bool(value["monotonic_non_decreasing"]) for value in cell_trends.values()
        ),
        "split_mean_deltas": split_means,
        "seed_mean_deltas": {str(key): value for key, value in seed_means.items()},
        "delta_variance_decomposition": {
            "total_ss": total_ss,
            "split_ss": split_ss,
            "seed_ss": seed_ss,
            "residual_split_by_seed_ss": max(0.0, total_ss - split_ss - seed_ss),
        },
    }


def _run_mapping(run: RelationExperimentRun) -> dict[str, Any]:
    marker = json.loads(
        (Path(run.output_dir) / "completed.json").read_text(encoding="utf-8")
    )
    return {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "seed": run.seed,
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "checkpoint_path": marker["checkpoint"],
    }


def _render_readme(
    summary_rows: Sequence[Mapping[str, Any]], trend: Mapping[str, Any]
) -> str:
    lines = [
        "# Relation-controlled zero-overlap CKA",
        "",
        "This 3 x 3 factorial study uses three low-correlation task selections ",
        "and three model seeds. Each selection contains exactly one member from ",
        "each predefined natural-dual pair, so no known dual is co-selected. ",
        "Every equal-k A-vs-B comparison has zero task-name overlap.",
        "",
        "| k | Final-layer linear CKA, mean +/- sample SD | Min | Max |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['trained_task_count']} | "
            f"{float(row['final_layer_linear_cka_mean']):.6f} +/- "
            f"{float(row['final_layer_linear_cka_sample_sd']):.6f} | "
            f"{float(row['final_layer_linear_cka_min']):.6f} | "
            f"{float(row['final_layer_linear_cka_max']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Mean paired k=8 minus k=1 delta: "
            f"{float(trend['delta_k8_minus_k1_mean']):+.6f} +/- "
            f"{float(trend['delta_k8_minus_k1_sample_sd']):.6f}.",
            f"Cells with positive paired delta: {trend['positive_delta_cells']}/9.",
            f"Two-sided exact sign-test p: "
            f"{float(trend['two_sided_exact_sign_test_p']):.6f}.",
            f"Strictly monotonic cells: {trend['monotonic_cell_count']}/9.",
            "",
            "The selection correlation screen used only the first 50,000 records ",
            "of training shard 000 and controlled permutation length. CKA uses the ",
            "same 4,096 task-free validation prefixes for every model. Test data ",
            "were not used.",
            "",
            "[Per-cell curves](cell_curves.csv) and "
            "[all layerwise comparisons](layerwise_cka.csv) are provided.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def run_relation_cka(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    matrix = relation_summary(config_path)
    if matrix["complete_count"] != matrix["run_count"]:
        raise ValueError("all 72 relation-controlled models must complete before CKA")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    probe_count = int(config["analysis"]["probe_examples"])
    probe_seed = int(config["analysis"]["probe_seed"])
    validation_manifest = Path(config["validation_manifest"])
    examples = select_probe_examples(
        validation_manifest, count=probe_count, seed=probe_seed, shard_index=0
    )
    probe = probe_identity(
        examples,
        dataset_manifest_sha256=_sha256(validation_manifest),
        shard_index=0,
        seed=probe_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(probe, output_dir / "probe_manifest.json")

    runs = build_relation_matrix(config_path)
    repository = Path.cwd()
    cache_dir = output_dir / "cache"
    activations: dict[str, Any] = {}
    for index, run in enumerate(runs, start=1):
        print(f"extracting relation activation {index}/{len(runs)}: {run.run_id}")
        activations[run.run_id] = _load_trained_activations(
            _run_mapping(run),
            repository=repository,
            examples=examples,
            probe_sha256=probe["probe_sha256"],
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
        )

    layer_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for split_id in SPLIT_IDS:
        for seed in MODEL_SEEDS:
            cell = [run for run in runs if run.split_id == split_id and run.seed == seed]
            by_key = {(run.pool, run.task_count): run for run in cell}
            for task_count in TASK_COUNTS:
                run_a = by_key["a", task_count]
                run_b = by_key["b", task_count]
                a = activations[run_a.run_id]
                b = activations[run_b.run_id]
                if tuple(a.layers) != tuple(b.layers):
                    raise ValueError("relation-controlled model layer grids differ")
                for layer in a.layers:
                    value = float(
                        linear_cka(a.layers[layer].to(device), b.layers[layer].to(device)).cpu()
                    )
                    if not math.isfinite(value):
                        raise ValueError("relation-controlled CKA is not finite")
                    layer_rows.append(
                        {
                            "split_id": split_id,
                            "model_seed": seed,
                            "trained_task_count": task_count,
                            "layer": layer,
                            "probe_examples": probe_count,
                            "linear_cka": value,
                            "run_id_a": run_a.run_id,
                            "run_id_b": run_b.run_id,
                        }
                    )
                    if layer == "final_norm":
                        cell_rows.append(
                            {
                                "split_id": split_id,
                                "model_seed": seed,
                                "trained_task_count": task_count,
                                "final_layer_linear_cka": value,
                            }
                        )
    summary_rows, trend = summarize_cell_curves(cell_rows)
    paths = {
        "cell_curves.csv": (cell_rows, CELL_FIELDS),
        "cka_summary.csv": (summary_rows, SUMMARY_FIELDS),
        "layerwise_cka.csv": (layer_rows, LAYER_FIELDS),
    }
    for name, (rows, fields) in paths.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme_path = output_dir / "README.md"
    _atomic_text(_render_readme(summary_rows, trend), readme_path)
    manifest = {
        "status": "completed",
        "protocol_version": "property32-relation-controlled-cka/v1",
        "analysis_commit": _git_commit(repository),
        "config_path": str(config_path),
        "config_sha256": matrix["config_sha256"],
        "run_count": len(runs),
        "cell_count": 9,
        "probe": probe,
        "trend": trend,
        "test_split_used": False,
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in paths},
            "README.md": _sha256(readme_path),
            "probe_manifest.json": _sha256(output_dir / "probe_manifest.json"),
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    result = run_relation_cka(
        args.config, args.output_dir, device=device, batch_size=args.batch_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "main",
    "run_relation_cka",
    "summarize_cell_curves",
]

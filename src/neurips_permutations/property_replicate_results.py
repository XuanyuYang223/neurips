"""Run and aggregate the three-replicate zero-overlap property analyses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .cka import _atomic_csv, _atomic_json, _atomic_text, _git_commit, _sha256
from .property_cka import run_property_cka, summarize_primary_trend
from .property_experiments import matrix_summary
from .property_replicates import REPLICATE_CONFIGS, validate_replicate_design
from .property_report import run_property_report


DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/replicates")
TASK_COUNTS = (1, 2, 4, 8, 16)
BEHAVIOR_METRICS = (
    "macro_loss",
    "macro_token_accuracy",
    "macro_sequence_accuracy",
    "macro_majority_baseline_sequence_accuracy",
    "macro_sequence_accuracy_minus_majority",
)

BEHAVIOR_REPLICATE_FIELDS = (
    "replicate_id",
    "model_seed",
    "trained_task_count",
    *BEHAVIOR_METRICS,
)
BEHAVIOR_SUMMARY_FIELDS = (
    "trained_task_count",
    "replicate_count",
    *tuple(
        name
        for metric in BEHAVIOR_METRICS
        for name in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)
CKA_REPLICATE_FIELDS = (
    "replicate_id",
    "model_seed",
    "trained_task_count",
    "final_layer_linear_cka",
)
CKA_SUMMARY_FIELDS = (
    "trained_task_count",
    "replicate_count",
    "final_layer_linear_cka_mean",
    "final_layer_linear_cka_sample_sd",
    "final_layer_linear_cka_min",
    "final_layer_linear_cka_max",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_behavior_replicate_rows(
    replicate_dirs: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Average A->B and B->A within each replicate before cross-replicate use."""

    rows: list[dict[str, Any]] = []
    for replicate_id, directory in replicate_dirs.items():
        config = tomllib.loads(
            REPLICATE_CONFIGS[replicate_id].read_text(encoding="utf-8")
        )
        model_seed = int(config["model_seeds"][0])
        source = _read_csv(directory / "behavior" / "SUMMARY.csv")
        for task_count in TASK_COUNTS:
            selected = [
                row
                for row in source
                if row["task_status"] == "opposite_pool"
                and int(row["trained_task_count"]) == task_count
            ]
            if len(selected) != 2 or {row["pool"] for row in selected} != {"A", "B"}:
                raise ValueError(
                    f"{replicate_id} k={task_count} lacks both opposite-pool directions"
                )
            row: dict[str, Any] = {
                "replicate_id": replicate_id,
                "model_seed": model_seed,
                "trained_task_count": task_count,
            }
            for metric in BEHAVIOR_METRICS:
                row[metric] = statistics.fmean(float(item[metric]) for item in selected)
            rows.append(row)
    if len(rows) != len(replicate_dirs) * len(TASK_COUNTS):
        raise ValueError("behavior replicate grid is incomplete")
    return rows


def summarize_behavior_replicates(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replicate_ids = {str(row["replicate_id"]) for row in rows}
    for task_count in TASK_COUNTS:
        selected = [
            row for row in rows if int(row["trained_task_count"]) == task_count
        ]
        if {str(row["replicate_id"]) for row in selected} != replicate_ids:
            raise ValueError(f"behavior k={task_count} has an incomplete replicate grid")
        summary: dict[str, Any] = {
            "trained_task_count": task_count,
            "replicate_count": len(selected),
        }
        for metric in BEHAVIOR_METRICS:
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(summary)
    return result


def build_cka_replicate_rows(
    replicate_dirs: Mapping[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate_id, directory in replicate_dirs.items():
        config = tomllib.loads(
            REPLICATE_CONFIGS[replicate_id].read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (directory / "cka" / "manifest.json").read_text(encoding="utf-8")
        )
        trend = manifest["primary_trend"]
        if tuple(trend["task_counts"]) != TASK_COUNTS:
            raise ValueError(f"{replicate_id} CKA task-count grid differs")
        for task_count, value in zip(
            trend["task_counts"], trend["final_layer_linear_cka"]
        ):
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "model_seed": int(config["model_seeds"][0]),
                    "trained_task_count": int(task_count),
                    "final_layer_linear_cka": float(value),
                }
            )
    if len(rows) != len(replicate_dirs) * len(TASK_COUNTS):
        raise ValueError("CKA replicate grid is incomplete")
    return rows


def summarize_cka_replicates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replicate_ids = {str(row["replicate_id"]) for row in rows}
    result: list[dict[str, Any]] = []
    mean_trend_rows: list[dict[str, Any]] = []
    for task_count in TASK_COUNTS:
        selected = [
            row for row in rows if int(row["trained_task_count"]) == task_count
        ]
        if {str(row["replicate_id"]) for row in selected} != replicate_ids:
            raise ValueError(f"CKA k={task_count} has an incomplete replicate grid")
        values = [float(row["final_layer_linear_cka"]) for row in selected]
        mean = statistics.fmean(values)
        result.append(
            {
                "trained_task_count": task_count,
                "replicate_count": len(values),
                "final_layer_linear_cka_mean": mean,
                "final_layer_linear_cka_sample_sd": statistics.stdev(values),
                "final_layer_linear_cka_min": min(values),
                "final_layer_linear_cka_max": max(values),
            }
        )
        mean_trend_rows.append(
            {
                "comparison": "disjoint_pools_equal_k",
                "layer": "final_norm",
                "task_count_a": task_count,
                "linear_cka": mean,
            }
        )
    trend = summarize_primary_trend(mean_trend_rows)
    peaks: dict[str, int] = {}
    per_replicate: dict[str, dict[str, Any]] = {}
    for replicate_id in sorted(replicate_ids):
        selected = sorted(
            (row for row in rows if row["replicate_id"] == replicate_id),
            key=lambda row: int(row["trained_task_count"]),
        )
        values = [float(row["final_layer_linear_cka"]) for row in selected]
        peak_k = TASK_COUNTS[max(range(len(values)), key=values.__getitem__)]
        peaks[replicate_id] = peak_k
        per_replicate[replicate_id] = summarize_primary_trend(
            [
                {
                    "comparison": "disjoint_pools_equal_k",
                    "layer": "final_norm",
                    "task_count_a": task_count,
                    "linear_cka": value,
                }
                for task_count, value in zip(TASK_COUNTS, values)
            ]
        )
    trend["replicate_peak_k"] = peaks
    trend["k8_peak_replicate_count"] = sum(value == 8 for value in peaks.values())
    trend["per_replicate"] = per_replicate
    return result, trend


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_readme(
    behavior: Sequence[Mapping[str, Any]],
    cka: Sequence[Mapping[str, Any]],
    trend: Mapping[str, Any],
) -> str:
    lines = [
        "# Three-replicate zero-overlap property results",
        "",
        "The study contains 30 independently trained Transformers: three joint ",
        "task-split/model-seed replicates, two disjoint pools, and five values of ",
        "`k`. Values are mean plus/minus sample standard deviation across the three ",
        "replicate-level measurements. Test data were not used.",
        "",
        "## Opposite-pool behavior",
        "",
        "| k | Loss | Token accuracy | Exact accuracy | Exact minus majority |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in behavior:
        lines.append(
            f"| {row['trained_task_count']} | "
            f"{float(row['macro_loss_mean']):.4f} +/- {float(row['macro_loss_sample_sd']):.4f} | "
            f"{_pct(float(row['macro_token_accuracy_mean']))} +/- {_pct(float(row['macro_token_accuracy_sample_sd']))} | "
            f"{_pct(float(row['macro_sequence_accuracy_mean']))} +/- {_pct(float(row['macro_sequence_accuracy_sample_sd']))} | "
            f"{100.0 * float(row['macro_sequence_accuracy_minus_majority_mean']):+.2f} +/- "
            f"{100.0 * float(row['macro_sequence_accuracy_minus_majority_sample_sd']):.2f} pp |"
        )
    lines.extend(
        [
            "",
            "Exact unseen-property accuracy remains below the task-specific ",
            "majority baseline at every k. The behavioral result therefore does ",
            "not demonstrate reliable hard zero-shot execution.",
            "",
            "## Final-layer A-vs-B linear CKA",
            "",
            "| k | Mean +/- sample SD | Min | Max |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in cka:
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
            f"Mean-trend Spearman rho: {float(trend['spearman_rho']):.6f}.",
            f"Mean-trend Pearson r against log2(k): "
            f"{float(trend['pearson_r_log2_k']):.6f}.",
            f"Replicates peaking at k=8: {trend['k8_peak_replicate_count']}/3.",
            f"Mean CKA is monotonic non-decreasing: "
            f"{str(bool(trend['monotonic_non_decreasing'])).lower()}.",
            "",
            "The three replicates jointly vary model seed and task split. Their ",
            "sample SD therefore captures combined variability and does not separate ",
            "the two variance sources. The fixed-update per-task exposure confound ",
            "also remains. See the ",
            "[frozen protocol](../../../PROPERTY32_REPLICATES.md) for details.",
            "",
            "## Artifacts",
            "",
            "- [Replicate-level behavioral values](behavior_replicates.csv)",
            "- [Behavioral mean and sample SD](behavior_summary.csv)",
            "- [Replicate-level CKA values](cka_replicates.csv)",
            "- [CKA mean, sample SD, minimum, and maximum](cka_summary.csv)",
            "- Child reports: [R0](r0/behavior/README.md), "
            "[R1](r1/behavior/README.md), and [R2](r2/behavior/README.md)",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def run_replicate_results(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    probe_count: int = 4_096,
    probe_seed: int = 20_260_902,
    batch_size: int = 256,
    device: torch.device,
) -> dict[str, Any]:
    design = validate_replicate_design()
    for replicate_id, config_path in REPLICATE_CONFIGS.items():
        status = matrix_summary(config_path)
        if status["complete_count"] != status["run_count"]:
            raise ValueError(f"{replicate_id} training matrix is incomplete")
    replicate_dirs = {
        replicate_id: output_dir / replicate_id for replicate_id in REPLICATE_CONFIGS
    }
    for replicate_id, config_path in REPLICATE_CONFIGS.items():
        run_property_report(config_path, replicate_dirs[replicate_id] / "behavior")
        run_property_cka(
            config_path,
            replicate_dirs[replicate_id] / "cka",
            probe_count=probe_count,
            probe_seed=probe_seed,
            batch_size=batch_size,
            device=device,
        )
    behavior_rows = build_behavior_replicate_rows(replicate_dirs)
    behavior_summary = summarize_behavior_replicates(behavior_rows)
    cka_rows = build_cka_replicate_rows(replicate_dirs)
    cka_summary, trend = summarize_cka_replicates(cka_rows)
    paths = {
        "behavior_replicates.csv": (behavior_rows, BEHAVIOR_REPLICATE_FIELDS),
        "behavior_summary.csv": (behavior_summary, BEHAVIOR_SUMMARY_FIELDS),
        "cka_replicates.csv": (cka_rows, CKA_REPLICATE_FIELDS),
        "cka_summary.csv": (cka_summary, CKA_SUMMARY_FIELDS),
    }
    for name, (rows, fields) in paths.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme_path = output_dir / "README.md"
    _atomic_text(_render_readme(behavior_summary, cka_summary, trend), readme_path)
    child_manifests = {
        f"{replicate_id}/{kind}/manifest.json": _sha256(
            directory / kind / "manifest.json"
        )
        for replicate_id, directory in replicate_dirs.items()
        for kind in ("behavior", "cka")
    }
    manifest = {
        "status": "completed",
        "protocol_version": "property32-zero-overlap-replicates/v1",
        "analysis_commit": _git_commit(Path.cwd()),
        "replicate_count": 3,
        "run_count": 30,
        "design": design,
        "mean_cka_trend": trend,
        "test_split_used": False,
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in paths},
            "README.md": _sha256(readme_path),
            **child_manifests,
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-count", type=int, default=4_096)
    parser.add_argument("--probe-seed", type=int, default=20_260_902)
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
    result = run_replicate_results(
        args.output_dir,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BEHAVIOR_METRICS",
    "DEFAULT_OUTPUT_DIR",
    "build_behavior_replicate_rows",
    "build_cka_replicate_rows",
    "main",
    "run_replicate_results",
    "summarize_behavior_replicates",
    "summarize_cka_replicates",
]

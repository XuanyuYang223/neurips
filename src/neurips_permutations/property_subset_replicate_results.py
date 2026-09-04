"""Analyze fixed-seed Property32 task-subset replicates R0, R3, and R4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch

from .cka import _atomic_csv, _atomic_json, _atomic_text, _git_commit, _sha256
from .property_cka import run_property_cka
from .property_experiments import matrix_summary
from .property_replicate_results import (
    BEHAVIOR_METRICS,
    BEHAVIOR_REPLICATE_FIELDS,
    BEHAVIOR_SUMMARY_FIELDS,
    CKA_REPLICATE_FIELDS,
    CKA_SUMMARY_FIELDS,
    summarize_behavior_replicates,
    summarize_cka_replicates,
)
from .property_report import run_property_report


DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/subset-replicates")
TASK_COUNTS = (1, 2, 4, 8, 16)
CONFIGS = {
    "r0": Path("configs/property32_zero_overlap_pilot.toml"),
    "r3": Path("configs/property32_zero_overlap_r3.toml"),
    "r4": Path("configs/property32_zero_overlap_r4.toml"),
}
R0_RESULTS = Path("results/property32-zero-overlap/replicates/r0")


def result_directories(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    return {"r0": R0_RESULTS, "r3": output_dir / "r3", "r4": output_dir / "r4"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_behavior_rows(directories: Mapping[str, Path]) -> list[dict[str, Any]]:
    """Average A-to-B and B-to-A within each split replicate and k."""

    rows = []
    for replicate_id in CONFIGS:
        source = _read_csv(directories[replicate_id] / "behavior" / "SUMMARY.csv")
        for task_count in TASK_COUNTS:
            selected = [
                row
                for row in source
                if row["task_status"] == "opposite_pool"
                and int(row["trained_task_count"]) == task_count
            ]
            if len(selected) != 2 or {row["pool"] for row in selected} != {"A", "B"}:
                raise ValueError(f"{replicate_id} k={task_count} lacks both directions")
            result: dict[str, Any] = {
                "replicate_id": replicate_id,
                "model_seed": 17,
                "trained_task_count": task_count,
            }
            for metric in BEHAVIOR_METRICS:
                result[metric] = statistics.fmean(float(row[metric]) for row in selected)
            rows.append(result)
    if len(rows) != 15:
        raise ValueError("fixed-seed behavior grid is incomplete")
    return rows


def build_cka_rows(directories: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for replicate_id in CONFIGS:
        manifest = json.loads(
            (directories[replicate_id] / "cka" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        trend = manifest["primary_trend"]
        if tuple(trend["task_counts"]) != TASK_COUNTS:
            raise ValueError(f"{replicate_id} CKA grid differs from the protocol")
        for task_count, value in zip(
            trend["task_counts"], trend["final_layer_linear_cka"], strict=True
        ):
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "model_seed": 17,
                    "trained_task_count": int(task_count),
                    "final_layer_linear_cka": float(value),
                }
            )
    if len(rows) != 15:
        raise ValueError("fixed-seed CKA grid is incomplete")
    return rows


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _readme(
    behavior: Sequence[Mapping[str, Any]],
    cka: Sequence[Mapping[str, Any]],
    trend: Mapping[str, Any],
) -> str:
    lines = [
        "# Fixed-seed task-subset replicate results",
        "",
        "R0, R3, and R4 all use Transformer seed 17 while independently changing the family-balanced A/B task partition. The error bars therefore estimate task-subset sensitivity without mixing it with initialization-seed variability.",
        "",
        "## Opposite-pool behavioral generalization",
        "",
        "| k | Loss | Token accuracy | Exact accuracy | Exact minus majority |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in behavior:
        lines.append(
            f"| {row['trained_task_count']} | "
            f"{float(row['macro_loss_mean']):.4f} ± {float(row['macro_loss_sample_sd']):.4f} | "
            f"{_pct(float(row['macro_token_accuracy_mean']))} ± {_pct(float(row['macro_token_accuracy_sample_sd']))} | "
            f"{_pct(float(row['macro_sequence_accuracy_mean']))} ± {_pct(float(row['macro_sequence_accuracy_sample_sd']))} | "
            f"{100*float(row['macro_sequence_accuracy_minus_majority_mean']):+.2f} ± "
            f"{100*float(row['macro_sequence_accuracy_minus_majority_sample_sd']):.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Final-layer linear CKA between disjoint pools",
            "",
            "| k | Mean ± sample SD | Min | Max |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in cka:
        lines.append(
            f"| {row['trained_task_count']} | "
            f"{float(row['final_layer_linear_cka_mean']):.6f} ± {float(row['final_layer_linear_cka_sample_sd']):.6f} | "
            f"{float(row['final_layer_linear_cka_min']):.6f} | {float(row['final_layer_linear_cka_max']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Mean-trend Spearman rho is {float(trend['spearman_rho']):.3f}; "
            f"the k=16 minus k=1 CKA change is {float(trend['delta_k16_minus_k1']):+.3f}.",
            "",
            "These three measurements are task-split replicates, not three independent random initializations. They should be reported separately from R0/R1/R2, which jointly varied both split and model seed.",
            "Validation data were used; the test split was not read.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_children(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    device: torch.device,
    probe_count: int = 4096,
    probe_seed: int = 20_260_902,
    batch_size: int = 256,
) -> None:
    directories = result_directories(output_dir)
    for replicate_id, config in CONFIGS.items():
        state = matrix_summary(config)
        if state["complete_count"] != state["run_count"]:
            raise ValueError(f"{replicate_id} training matrix is incomplete")
        if replicate_id == "r0":
            for kind in ("behavior", "cka"):
                if not (directories[replicate_id] / kind / "manifest.json").is_file():
                    raise ValueError(f"the frozen R0 {kind} artifact is missing")
            continue
        run_property_report(config, directories[replicate_id] / "behavior")
        run_property_cka(
            config,
            directories[replicate_id] / "cka",
            probe_count=probe_count,
            probe_seed=probe_seed,
            batch_size=batch_size,
            device=device,
        )


def aggregate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    directories = result_directories(output_dir)
    behavior_rows = build_behavior_rows(directories)
    behavior_summary = summarize_behavior_replicates(behavior_rows)
    cka_rows = build_cka_rows(directories)
    cka_summary, trend = summarize_cka_replicates(cka_rows)
    paths = {
        "behavior_replicates.csv": (behavior_rows, BEHAVIOR_REPLICATE_FIELDS),
        "behavior_summary.csv": (behavior_summary, BEHAVIOR_SUMMARY_FIELDS),
        "cka_replicates.csv": (cka_rows, CKA_REPLICATE_FIELDS),
        "cka_summary.csv": (cka_summary, CKA_SUMMARY_FIELDS),
    }
    for name, (rows, fields) in paths.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme = output_dir / "README.md"
    _atomic_text(_readme(behavior_summary, cka_summary, trend), readme)
    child_hashes = {
        f"{replicate_id}/{kind}/manifest.json": _sha256(
            directory / kind / "manifest.json"
        )
        for replicate_id, directory in directories.items()
        for kind in ("behavior", "cka")
    }
    manifest = {
        "status": "completed",
        "protocol_version": "property32-fixed-seed-subset-replicates/v1",
        "analysis_commit": _git_commit(Path.cwd()),
        "replicate_ids": list(CONFIGS),
        "model_seed": 17,
        "run_count": 30,
        "test_split_used": False,
        "mean_cka_trend": trend,
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in paths},
            "README.md": _sha256(readme),
            **child_hashes,
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    device: torch.device,
    probe_count: int = 4096,
    probe_seed: int = 20_260_902,
    batch_size: int = 256,
) -> dict[str, Any]:
    analyze_children(
        output_dir,
        device=device,
        probe_count=probe_count,
        probe_seed=probe_seed,
        batch_size=batch_size,
    )
    return aggregate(output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-count", type=int, default=4096)
    parser.add_argument("--probe-seed", type=int, default=20_260_902)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    result = run(
        args.output_dir,
        device=device,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONFIGS",
    "aggregate",
    "analyze_children",
    "build_behavior_rows",
    "build_cka_rows",
    "main",
    "result_directories",
    "run",
]

"""Deterministic behavioral report for the 32-property zero-overlap pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

from .cka import _atomic_csv, _atomic_json, _atomic_text, _git_commit, _sha256
from .math_ops import PROPERTY32_TASK_NAMES
from .property_experiments import (
    DEFAULT_CONFIG,
    PropertyExperimentRun,
    build_property_matrix,
    matrix_summary,
)


DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/behavior")

LOCAL_TASKS = frozenset(
    {
        "descents",
        "recoils",
        "peaks",
        "valleys",
        "double_ascents",
        "double_descents",
        "successions",
        "adjacencies",
    }
)
POSITIONAL_TASKS = frozenset(
    {
        "fixed_points",
        "anti_fixed_points",
        "exceedances",
        "deficiencies",
        "left_to_right_maxima",
        "left_to_right_minima",
        "right_to_left_maxima",
        "right_to_left_minima",
    }
)
CYCLE_TASKS = frozenset(
    {
        "cycle_count",
        "two_cycle_count",
        "three_cycle_count",
        "even_cycle_count",
        "odd_cycle_count",
        "longest_cycle",
        "shortest_cycle",
        "nontrivial_cycle_count",
    }
)
GLOBAL_RUN_TASKS = frozenset(PROPERTY32_TASK_NAMES) - (
    LOCAL_TASKS | POSITIONAL_TASKS | CYCLE_TASKS
)
TASK_FAMILY = {
    **{task: "local" for task in LOCAL_TASKS},
    **{task: "positional" for task in POSITIONAL_TASKS},
    **{task: "cycle" for task in CYCLE_TASKS},
    **{task: "global_run" for task in GLOBAL_RUN_TASKS},
}
if set(TASK_FAMILY) != set(PROPERTY32_TASK_NAMES):  # pragma: no cover
    raise RuntimeError("property family registry does not cover all 32 tasks")

RAW_FIELDS = (
    "pool",
    "trained_task_count",
    "run_id",
    "architecture",
    "seed",
    "task",
    "task_family",
    "task_status",
    "evaluation_split",
    "examples",
    "supervised_tokens",
    "loss",
    "token_accuracy",
    "sequence_accuracy",
)

SUMMARY_FIELDS = (
    "pool",
    "trained_task_count",
    "task_status",
    "task_count",
    "macro_loss",
    "macro_token_accuracy",
    "macro_sequence_accuracy",
)


def task_status(
    run: PropertyExperimentRun,
    task: str,
    *,
    pool_a: Sequence[str],
    pool_b: Sequence[str],
) -> str:
    """Classify a task relative to one independently trained checkpoint."""

    if task in run.tasks:
        return "seen"
    own_pool = pool_a if run.pool == "a" else pool_b
    other_pool = pool_b if run.pool == "a" else pool_a
    if task in other_pool:
        return "opposite_pool"
    if task in own_pool:
        return "same_pool_unseen"
    raise ValueError(f"task {task!r} is outside the frozen pools")


def build_raw_rows(
    runs: Sequence[PropertyExperimentRun],
    *,
    pool_a: Sequence[str],
    pool_b: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        marker_path = Path(run.output_dir) / "completed.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        validation = marker.get("validation")
        if not isinstance(validation, dict) or set(validation) != set(
            PROPERTY32_TASK_NAMES
        ):
            raise ValueError(f"{run.run_id} does not contain all 32 task metrics")
        for task in PROPERTY32_TASK_NAMES:
            metric = validation[task]
            rows.append(
                {
                    "pool": run.pool.upper(),
                    "trained_task_count": run.task_count,
                    "run_id": run.run_id,
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "task": task,
                    "task_family": TASK_FAMILY[task],
                    "task_status": task_status(
                        run, task, pool_a=pool_a, pool_b=pool_b
                    ),
                    "evaluation_split": "validation",
                    "examples": int(metric["examples"]),
                    "supervised_tokens": int(metric["tokens"]),
                    "loss": float(metric["loss"]),
                    "token_accuracy": float(metric["token_accuracy"]),
                    "sequence_accuracy": float(metric["sequence_accuracy"]),
                }
            )
    if len(rows) != len(runs) * 32:
        raise ValueError("raw report row count is incomplete")
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["pool"]),
            int(row["trained_task_count"]),
            str(row["task_status"]),
        )
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    status_order = {"seen": 0, "same_pool_unseen": 1, "opposite_pool": 2}
    for (pool, task_count, status), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            status_order[item[0][2]],
        ),
    ):
        result.append(
            {
                "pool": pool,
                "trained_task_count": task_count,
                "task_status": status,
                "task_count": len(values),
                "macro_loss": statistics.fmean(float(row["loss"]) for row in values),
                "macro_token_accuracy": statistics.fmean(
                    float(row["token_accuracy"]) for row in values
                ),
                "macro_sequence_accuracy": statistics.fmean(
                    float(row["sequence_accuracy"]) for row in values
                ),
            }
        )
    return result


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_readme(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lookup = {
        (str(row["pool"]), int(row["trained_task_count"]), str(row["task_status"])): row
        for row in summary_rows
    }
    lines = [
        "# Behavioral results: zero-overlap 32-property pilot",
        "",
        "Ten independently trained Transformers are compared: Pool A and Pool B ",
        "at `k = 1, 2, 4, 8, 16`. A model's opposite pool has no task overlap ",
        "with its training set. All metrics below are task-macro averages on the ",
        "diagnostic validation split; the held-back test split was not used.",
        "",
        "## Opposite-pool transfer",
        "",
        "| Pool | k | Loss | Token accuracy | Exact-sequence accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for pool in ("A", "B"):
        for task_count in (1, 2, 4, 8, 16):
            row = lookup[pool, task_count, "opposite_pool"]
            lines.append(
                f"| {pool} | {task_count} | {float(row['macro_loss']):.4f} | "
                f"{_percent(float(row['macro_token_accuracy']))} | "
                f"{_percent(float(row['macro_sequence_accuracy']))} |"
            )
    lines.extend(
        [
            "",
            "## Seen-task performance",
            "",
            "| Pool | k | Loss | Token accuracy | Exact-sequence accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pool in ("A", "B"):
        for task_count in (1, 2, 4, 8, 16):
            row = lookup[pool, task_count, "seen"]
            lines.append(
                f"| {pool} | {task_count} | {float(row['macro_loss']):.4f} | "
                f"{_percent(float(row['macro_token_accuracy']))} | "
                f"{_percent(float(row['macro_sequence_accuracy']))} |"
            )
    lines.extend(
        [
            "",
            "`token_accuracy` is teacher-forced accuracy over the scalar answer token ",
            "and EOS. `sequence_accuracy` requires both tokens to be correct and is the ",
            "primary complete-answer metric. `MODEL_TASK_ACCURACIES.csv` contains every ",
            "unaveraged model-task result; `SUMMARY.csv` contains the task-macro values.",
            "",
            "This is a one-seed pilot with a fixed 20,000-update budget. Per-task ",
            "exposure therefore falls as k increases, so any trend mixes task diversity ",
            "with reduced examples per learned task. It is descriptive, not a ",
            "population-level claim with error bars.",
            "",
        ]
    )
    return "\n".join(lines)


def run_property_report(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    matrix = matrix_summary(config_path)
    if matrix["complete_count"] != matrix["run_count"]:
        raise ValueError("all property pilot models must complete before reporting")
    runs = build_property_matrix(config_path)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    rows = build_raw_rows(runs, pool_a=config["pool_a"], pool_b=config["pool_b"])
    summaries = summarize_rows(rows)
    raw_path = output_dir / "MODEL_TASK_ACCURACIES.csv"
    summary_path = output_dir / "SUMMARY.csv"
    readme_path = output_dir / "README.md"
    _atomic_csv(rows, RAW_FIELDS, raw_path)
    _atomic_csv(summaries, SUMMARY_FIELDS, summary_path)
    _atomic_text(_render_readme(summaries), readme_path)
    result = {
        "status": "completed",
        "protocol_version": "property32-zero-overlap-behavior/v1",
        "analysis_commit": _git_commit(Path.cwd()),
        "config_sha256": matrix["config_sha256"],
        "run_count": len(runs),
        "raw_row_count": len(rows),
        "test_split_used": False,
        "artifacts": {
            raw_path.name: _sha256(raw_path),
            summary_path.name: _sha256(summary_path),
            readme_path.name: _sha256(readme_path),
        },
    }
    _atomic_json(result, output_dir / "manifest.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_property_report(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "RAW_FIELDS",
    "SUMMARY_FIELDS",
    "build_raw_rows",
    "main",
    "run_property_report",
    "summarize_rows",
    "task_status",
]

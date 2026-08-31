"""Export Henry-style 20-shot test metrics and paired adaptation gains."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .fewshot import DEFAULT_CONFIG, TEST_FORMAT_VERSION, _sha256, audit_all, load_spec


RAW_FIELDS = (
    "run_id",
    "initialization",
    "architecture",
    "base_run_id",
    "base_trained_task_count",
    "seed",
    "task",
    "shots",
    "evaluation_split",
    "examples",
    "supervised_tokens",
    "loss",
    "token_accuracy",
    "sequence_accuracy",
    "zero_shot_loss",
    "zero_shot_token_accuracy",
    "zero_shot_sequence_accuracy",
    "loss_improvement_from_zero_shot",
    "token_accuracy_improvement_from_zero_shot",
    "sequence_accuracy_improvement_from_zero_shot",
    "checkpoint_sha256",
    "fewshot_config_sha256",
    "support_artifact_sha256",
    "test_manifest_sha256",
    "evaluator_commit",
)

SUMMARY_FIELDS = (
    "initialization",
    "architecture",
    "base_trained_task_count",
    "task_count",
    "seed_count",
    "loss_mean",
    "loss_sample_sd",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
)

TASK_SUMMARY_FIELDS = (
    "initialization",
    "architecture",
    "base_trained_task_count",
    "task",
    "seed_count",
    "loss_mean",
    "loss_sample_sd",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
)

GAIN_FIELDS = (
    "architecture",
    "base_trained_task_count",
    "task_count",
    "seed_count",
    "loss_improvement_from_zero_shot_mean",
    "loss_improvement_from_zero_shot_sample_sd",
    "token_accuracy_improvement_from_zero_shot_mean",
    "token_accuracy_improvement_from_zero_shot_sample_sd",
    "sequence_accuracy_improvement_from_zero_shot_mean",
    "sequence_accuracy_improvement_from_zero_shot_sample_sd",
    "loss_improvement_over_random_mean",
    "loss_improvement_over_random_sample_sd",
    "token_accuracy_improvement_over_random_mean",
    "token_accuracy_improvement_over_random_sample_sd",
    "sequence_accuracy_improvement_over_random_mean",
    "sequence_accuracy_improvement_over_random_sample_sd",
)


def _mean(values: Iterable[float]) -> float:
    result = statistics.fmean(values)
    if not isinstance(result, float):  # pragma: no cover
        raise TypeError
    return result


def load_rows(
    config_path: Path = DEFAULT_CONFIG,
    *,
    zero_shot_evaluation_dir: Path = Path("results/v3/evaluation"),
) -> list[dict[str, Any]]:
    spec = load_spec(config_path)
    audit = audit_all(config_path)
    if not audit["ok"]:
        raise ValueError("all few-shot checkpoints must pass audit before export")
    manifest_path = spec.evaluation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "format_version": TEST_FORMAT_VERSION,
        "status": "completed",
        "fewshot_config_sha256": spec.config_sha256,
        "support_artifact_sha256": _sha256(spec.support_artifact),
        "test_manifest_sha256": spec.test_manifest_sha256,
        "run_count": spec.expected_runs,
        "examples_per_run": spec.expected_test_examples,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("few-shot test manifest identity is invalid")
    zero_root = (
        zero_shot_evaluation_dir
        if zero_shot_evaluation_dir.is_absolute()
        else spec.repository / zero_shot_evaluation_dir
    )
    rows: list[dict[str, Any]] = []
    for entry in manifest["runs"]:
        result_path = spec.evaluation_dir / str(entry["result_file"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("run_id") != entry.get("run_id"):
            raise ValueError(f"invalid few-shot result: {result_path}")
        metric = result.get("metrics")
        if not isinstance(metric, Mapping) or metric.get("examples") != spec.expected_test_examples:
            raise ValueError(f"invalid few-shot metrics: {result_path}")
        row: dict[str, Any] = {
            "run_id": result["run_id"],
            "initialization": result["initialization"],
            "architecture": result["architecture"],
            "base_run_id": result["base_run_id"],
            "base_trained_task_count": result["base_trained_task_count"],
            "seed": result["seed"],
            "task": result["task"],
            "shots": spec.shots,
            "evaluation_split": "test_shard_099",
            "examples": metric["examples"],
            "supervised_tokens": metric["tokens"],
            "loss": metric["loss"],
            "token_accuracy": metric["token_accuracy"],
            "sequence_accuracy": metric["sequence_accuracy"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "fewshot_config_sha256": result["fewshot_config_sha256"],
            "support_artifact_sha256": result["support_artifact_sha256"],
            "test_manifest_sha256": result["test_manifest_sha256"],
            "evaluator_commit": result["evaluator_commit"],
        }
        if result["initialization"] == "pretrained":
            zero_path = zero_root / "per-run" / f"{result['base_run_id']}.json"
            zero = json.loads(zero_path.read_text(encoding="utf-8"))
            zero_metric = zero["metrics"][result["task"]]
            row.update(
                {
                    "zero_shot_loss": zero_metric["loss"],
                    "zero_shot_token_accuracy": zero_metric["token_accuracy"],
                    "zero_shot_sequence_accuracy": zero_metric["sequence_accuracy"],
                    "loss_improvement_from_zero_shot": float(zero_metric["loss"])
                    - float(metric["loss"]),
                    "token_accuracy_improvement_from_zero_shot": float(
                        metric["token_accuracy"]
                    )
                    - float(zero_metric["token_accuracy"]),
                    "sequence_accuracy_improvement_from_zero_shot": float(
                        metric["sequence_accuracy"]
                    )
                    - float(zero_metric["sequence_accuracy"]),
                }
            )
        else:
            row.update(
                {
                    "zero_shot_loss": "",
                    "zero_shot_token_accuracy": "",
                    "zero_shot_sequence_accuracy": "",
                    "loss_improvement_from_zero_shot": "",
                    "token_accuracy_improvement_from_zero_shot": "",
                    "sequence_accuracy_improvement_from_zero_shot": "",
                }
            )
        rows.append(row)
    if len(rows) != spec.expected_runs:
        raise ValueError("few-shot raw result row count is incomplete")
    return rows


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected = set(tasks) if tasks is not None else None
    per_seed: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if selected is not None and row["task"] not in selected:
            continue
        key = (
            str(row["initialization"]),
            str(row["architecture"]),
            int(row["base_trained_task_count"]),
            int(row["seed"]),
        )
        per_seed.setdefault(key, []).append(row)
    seed_macros: list[dict[str, Any]] = []
    for (initialization, architecture, task_count, seed), group in per_seed.items():
        expected_tasks = 4 if selected is None else len(selected)
        if len(group) != expected_tasks or len({row["task"] for row in group}) != expected_tasks:
            raise ValueError("each few-shot seed macro has the wrong task count")
        seed_macros.append(
            {
                "initialization": initialization,
                "architecture": architecture,
                "base_trained_task_count": task_count,
                "seed": seed,
                "task_count": expected_tasks,
                "loss": _mean(float(row["loss"]) for row in group),
                "token_accuracy": _mean(float(row["token_accuracy"]) for row in group),
                "sequence_accuracy": _mean(
                    float(row["sequence_accuracy"]) for row in group
                ),
            }
        )
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in seed_macros:
        grouped.setdefault(
            (
                str(row["initialization"]),
                str(row["architecture"]),
                int(row["base_trained_task_count"]),
            ),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (initialization, architecture, task_count), group in grouped.items():
        if len(group) != 3 or len({row["seed"] for row in group}) != 3:
            raise ValueError("few-shot summary group must contain three seeds")
        record: dict[str, Any] = {
            "initialization": initialization,
            "architecture": architecture,
            "base_trained_task_count": task_count,
            "task_count": int(group[0]["task_count"]),
            "seed_count": 3,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_sample_sd"] = statistics.stdev(values)
        output.append(record)
    return output


def build_task_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["initialization"]),
                str(row["architecture"]),
                int(row["base_trained_task_count"]),
                str(row["task"]),
            ),
            [],
        ).append(row)
    output: list[dict[str, Any]] = []
    for (initialization, architecture, task_count, task), group in grouped.items():
        if len(group) != 3 or len({row["seed"] for row in group}) != 3:
            raise ValueError("few-shot task summary must contain three seeds")
        record: dict[str, Any] = {
            "initialization": initialization,
            "architecture": architecture,
            "base_trained_task_count": task_count,
            "task": task,
            "seed_count": 3,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_sample_sd"] = statistics.stdev(values)
        output.append(record)
    return output


def build_gains(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    random_rows = {
        (row["architecture"], int(row["seed"]), row["task"]): row
        for row in rows
        if row["initialization"] == "random"
    }
    pretrained = [row for row in rows if row["initialization"] == "pretrained"]
    per_seed: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in pretrained:
        per_seed.setdefault(
            (
                str(row["architecture"]),
                int(row["base_trained_task_count"]),
                int(row["seed"]),
            ),
            [],
        ).append(row)
    macros: list[dict[str, Any]] = []
    for (architecture, task_count, seed), group in per_seed.items():
        if len(group) != 4:
            raise ValueError("adaptation gain requires four paired holdout tasks")
        record: dict[str, Any] = {
            "architecture": architecture,
            "base_trained_task_count": task_count,
            "seed": seed,
            "task_count": 4,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            zero_key = f"{metric}_improvement_from_zero_shot"
            record[zero_key] = _mean(float(row[zero_key]) for row in group)
            if metric == "loss":
                record[f"{metric}_improvement_over_random"] = _mean(
                    float(random_rows[(architecture, seed, row["task"])][metric])
                    - float(row[metric])
                    for row in group
                )
            else:
                record[f"{metric}_improvement_over_random"] = _mean(
                    float(row[metric])
                    - float(random_rows[(architecture, seed, row["task"])][metric])
                    for row in group
                )
        macros.append(record)
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in macros:
        grouped.setdefault(
            (str(row["architecture"]), int(row["base_trained_task_count"])), []
        ).append(row)
    output = []
    for (architecture, task_count), group in grouped.items():
        if len(group) != 3:
            raise ValueError("gain summary must contain three paired seeds")
        record: dict[str, Any] = {
            "architecture": architecture,
            "base_trained_task_count": task_count,
            "task_count": 4,
            "seed_count": 3,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            for comparison in ("from_zero_shot", "over_random"):
                key = f"{metric}_improvement_{comparison}"
                values = [float(row[key]) for row in group]
                record[f"{key}_mean"] = statistics.fmean(values)
                record[f"{key}_sample_sd"] = statistics.stdev(values)
        output.append(record)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: format(row[field], ".17g")
                        if isinstance(row[field], float)
                        else row[field]
                        for field in fields
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def export_results(
    config_path: Path = DEFAULT_CONFIG,
    *,
    output_dir: Path = Path("results/v3/fewshot"),
    zero_shot_evaluation_dir: Path = Path("results/v3/evaluation"),
) -> dict[str, Any]:
    spec = load_spec(config_path)
    destination = output_dir if output_dir.is_absolute() else spec.repository / output_dir
    rows = load_rows(
        config_path, zero_shot_evaluation_dir=zero_shot_evaluation_dir
    )
    summary = build_summary(rows)
    structured_summary = build_summary(
        rows, tasks=("to_reduced_word", "compose", "to_lehmer")
    )
    task_summary = build_task_summary(rows)
    gains = build_gains(rows)
    outputs = {
        "raw": destination / "test_model_task_accuracies.csv",
        "summary": destination / "test_summary.csv",
        "structured_summary": destination / "test_structured_summary.csv",
        "task_summary": destination / "test_task_summary.csv",
        "gains": destination / "test_adaptation_gains.csv",
    }
    _write_csv(outputs["raw"], rows, RAW_FIELDS)
    _write_csv(outputs["summary"], summary, SUMMARY_FIELDS)
    _write_csv(outputs["structured_summary"], structured_summary, SUMMARY_FIELDS)
    _write_csv(outputs["task_summary"], task_summary, TASK_SUMMARY_FIELDS)
    _write_csv(outputs["gains"], gains, GAIN_FIELDS)
    return {
        "status": "completed",
        "raw_rows": len(rows),
        "summary_rows": len(summary),
        "structured_summary_rows": len(structured_summary),
        "task_summary_rows": len(task_summary),
        "gain_rows": len(gains),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v3/fewshot"))
    parser.add_argument(
        "--zero-shot-evaluation-dir",
        type=Path,
        default=Path("results/v3/evaluation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_results(
        args.config,
        output_dir=args.output_dir,
        zero_shot_evaluation_dir=args.zero_shot_evaluation_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

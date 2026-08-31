"""Export audited v3 validation metrics and generalization summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import statistics
import tempfile
import tomllib
from typing import Any, Iterable, Mapping, Sequence

from .audit import audit_experiment
from .experiments import build_experiment_matrix, task_names_for_experiment


DEFAULT_CONFIG = Path("configs/henry_permutation_revised.toml")

ENCODING_TASKS = {
    "to_cycle",
    "to_lehmer",
    "to_inversion_vector",
    "to_reduced_word",
}
ALGEBRA_TASKS = {
    "inverse",
    "compose",
    "right_multiply_simple",
    "bruhat_leq",
}

VALIDATION_FIELDS = (
    "protocol_version",
    "matrix",
    "condition",
    "run_id",
    "architecture",
    "trained_task_count",
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
    "checkpoint_sha256",
    "experiment_config_sha256",
    "evaluation_parent_manifest_sha256",
    "evaluation_shards",
    "validation_split_manifest_sha256",
)

RUN_SUMMARY_FIELDS = (
    "protocol_version",
    "matrix",
    "condition",
    "run_id",
    "architecture",
    "trained_task_count",
    "seed",
    "evaluation_group",
    "task_count",
    "task_macro_token_accuracy",
    "task_macro_sequence_accuracy",
)

NESTED_SUMMARY_FIELDS = (
    "architecture",
    "trained_task_count",
    "task_status",
    "seed_count",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
)

CATEGORY_SUMMARY_FIELDS = (
    "architecture",
    "training_condition",
    "evaluation_condition",
    "seed_count",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_family(task: str) -> str:
    if task in ENCODING_TASKS:
        return "encoding"
    if task in ALGEBRA_TASKS:
        return "algebra"
    return "statistics"


def _category_groups(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    comparison = config["category_comparison"]
    return {
        condition["name"]: tuple(condition["tasks"])
        for condition in comparison["conditions"]
    }


def _task_status(
    *,
    matrix: str,
    task: str,
    trained_tasks: set[str],
    holdouts: set[str],
    matched_category_tasks: set[str],
) -> str:
    if task in trained_tasks:
        return "seen"
    if matrix == "nested":
        return "fixed_train_holdout" if task in holdouts else "pool_unseen"
    if task in matched_category_tasks:
        return "cross_category_matched"
    return "other_unmatched_statistics"


def validation_rows_from_audits(
    config: Mapping[str, Any],
    audits: Mapping[str, Mapping[str, Any]],
    *,
    validation_split_manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Flatten two successful strict audits into 960 deterministic rows."""

    task_names = tuple(task_names_for_experiment(dict(config)))
    holdouts = set(config["holdout_tasks"])
    category_groups = _category_groups(config)
    matched_category_tasks = {
        task for tasks in category_groups.values() for task in tasks
    }
    expected_runs = {"nested": 30, "category": 18}
    rows: list[dict[str, Any]] = []

    for matrix in ("nested", "category"):
        audit = audits[matrix]
        if not audit.get("ok"):
            raise ValueError(f"{matrix} strict audit did not pass")
        run_records = audit.get("runs")
        if not isinstance(run_records, list) or len(run_records) != expected_runs[matrix]:
            raise ValueError(f"{matrix} audit has an unexpected run count")
        for run in run_records:
            if run.get("status") != "passed" or run.get("results") is None:
                raise ValueError(f"run {run.get('run_id')} did not pass strict audit")
            validation = run["results"].get("validation")
            if not isinstance(validation, Mapping) or set(validation) != set(task_names):
                raise ValueError(f"run {run['run_id']} has an invalid validation grid")
            trained_tasks = set(run["tasks"])
            condition = run.get("condition", "")
            for task in task_names:
                metric = validation[task]
                row = {
                    "protocol_version": config["protocol_version"],
                    "matrix": matrix,
                    "condition": condition,
                    "run_id": run["run_id"],
                    "architecture": run["architecture"],
                    "trained_task_count": run["task_count"],
                    "seed": run["seed"],
                    "task": task,
                    "task_family": _task_family(task),
                    "task_status": _task_status(
                        matrix=matrix,
                        task=task,
                        trained_tasks=trained_tasks,
                        holdouts=holdouts,
                        matched_category_tasks=matched_category_tasks,
                    ),
                    "evaluation_split": "validation_shard_098",
                    "examples": metric["examples"],
                    "supervised_tokens": metric["tokens"],
                    "loss": metric["loss"],
                    "token_accuracy": metric["token_accuracy"],
                    "sequence_accuracy": metric["sequence_accuracy"],
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "experiment_config_sha256": audit["config_sha256"],
                    "evaluation_parent_manifest_sha256": audit["manifests"]["validation"]["sha256"],
                    "evaluation_shards": "098",
                    "validation_split_manifest_sha256": validation_split_manifest_sha256,
                }
                for key in ("loss", "token_accuracy", "sequence_accuracy"):
                    value = row[key]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"non-finite {key} in {run['run_id']}:{task}")
                rows.append(row)

    if len(rows) != 48 * len(task_names):
        raise ValueError(f"expected 960 validation rows, found {len(rows)}")
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty group")
    return statistics.fmean(materialized)


def build_run_summaries(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return task-macro metrics for every run and evaluation group."""

    category_groups = _category_groups(config)
    by_run: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_run.setdefault((str(row["matrix"]), str(row["run_id"])), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (matrix, _), run_rows in by_run.items():
        first = run_rows[0]
        if matrix == "nested":
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in run_rows:
                grouped.setdefault(str(row["task_status"]), []).append(row)
        else:
            grouped = {
                name: [row for row in run_rows if row["task"] in tasks]
                for name, tasks in category_groups.items()
            }
        for group, group_rows in grouped.items():
            if not group_rows:
                continue
            summaries.append(
                {
                    "protocol_version": first["protocol_version"],
                    "matrix": matrix,
                    "condition": first["condition"],
                    "run_id": first["run_id"],
                    "architecture": first["architecture"],
                    "trained_task_count": first["trained_task_count"],
                    "seed": first["seed"],
                    "evaluation_group": group,
                    "task_count": len(group_rows),
                    "task_macro_token_accuracy": _mean(
                        float(row["token_accuracy"]) for row in group_rows
                    ),
                    "task_macro_sequence_accuracy": _mean(
                        float(row["sequence_accuracy"]) for row in group_rows
                    ),
                }
            )
    return summaries


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in key_fields), []).append(row)
    output: list[dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) != 3 or len({row["seed"] for row in group}) != 3:
            raise ValueError(f"summary group {key!r} does not contain three seeds")
        token = [float(row["task_macro_token_accuracy"]) for row in group]
        sequence = [float(row["task_macro_sequence_accuracy"]) for row in group]
        record = dict(zip(key_fields, key, strict=True))
        record.update(
            {
                "seed_count": 3,
                "token_accuracy_mean": statistics.fmean(token),
                "token_accuracy_sample_sd": statistics.stdev(token),
                "sequence_accuracy_mean": statistics.fmean(sequence),
                "sequence_accuracy_sample_sd": statistics.stdev(sequence),
            }
        )
        output.append(record)
    return output


def build_nested_summary(run_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in run_summaries if row["matrix"] == "nested"]
    renamed = [dict(row, task_status=row["evaluation_group"]) for row in rows]
    return _aggregate_seed_rows(
        renamed,
        ("architecture", "trained_task_count", "task_status"),
    )


def build_category_summary(run_summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        dict(
            row,
            training_condition=row["condition"],
            evaluation_condition=row["evaluation_group"],
        )
        for row in run_summaries
        if row["matrix"] == "category"
    ]
    return _aggregate_seed_rows(
        rows,
        ("architecture", "training_condition", "evaluation_condition"),
    )


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _format_csv_value(row[key]) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def export_results(config_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = config_path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    # Validate the frozen run definitions before trusting audit output order.
    for matrix, expected in (("nested", 30), ("category", 18)):
        if len(build_experiment_matrix(config_path, matrix=matrix)) != expected:
            raise ValueError(f"{matrix} run matrix has changed")
    audits = {
        matrix: audit_experiment(config_path, matrix=matrix)
        for matrix in ("nested", "category")
    }
    validation_split_manifest = Path(config["dataset_manifest"]).parent / "validation_manifest.json"
    validation_rows = validation_rows_from_audits(
        config,
        audits,
        validation_split_manifest_sha256=_sha256(validation_split_manifest),
    )
    run_summaries = build_run_summaries(validation_rows, config)
    nested_summary = build_nested_summary(run_summaries)
    category_summary = build_category_summary(run_summaries)

    outputs = {
        "validation": output_dir / "V3_MODEL_TASK_ACCURACIES.csv",
        "runs": output_dir / "V3_RUN_SUMMARIES.csv",
        "nested": output_dir / "V3_NESTED_SUMMARY.csv",
        "category": output_dir / "V3_CATEGORY_SUMMARY.csv",
    }
    write_csv_atomic(outputs["validation"], validation_rows, VALIDATION_FIELDS)
    write_csv_atomic(outputs["runs"], run_summaries, RUN_SUMMARY_FIELDS)
    write_csv_atomic(outputs["nested"], nested_summary, NESTED_SUMMARY_FIELDS)
    write_csv_atomic(outputs["category"], category_summary, CATEGORY_SUMMARY_FIELDS)
    return {
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "validation_rows": len(validation_rows),
        "run_summary_rows": len(run_summaries),
        "nested_summary_rows": len(nested_summary),
        "category_summary_rows": len(category_summary),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_results(args.config, args.output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

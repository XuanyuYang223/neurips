"""Export audited v3 validation metrics and generalization summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
DEFAULT_OUTPUT_DIR = Path("results/v3")
DEFAULT_TEST_EVALUATION_DIR = Path("results/v3/evaluation")

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
    "matrix",
    "condition",
    "run_id",
    "architecture",
    "trained_task_count",
    "seed",
    "evaluation_group",
    "task_count",
    "task_macro_loss",
    "task_macro_token_accuracy",
    "task_macro_sequence_accuracy",
)

NESTED_SUMMARY_FIELDS = (
    "architecture",
    "trained_task_count",
    "task_status",
    "seed_count",
    "loss_mean",
    "loss_sample_sd",
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
    "loss_mean",
    "loss_sample_sd",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
)

TEST_FIELDS = (
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
    "test_manifest_sha256",
    "evaluation_shards",
    "evaluator_commit",
)

NESTED_GENERALIZATION_FIELDS = (
    "architecture",
    "trained_task_count",
    "evaluation_group",
    "evaluated_unseen_task_count",
    "seed_count",
    "loss_mean",
    "loss_sample_sd",
    "loss_change_from_k1",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "token_accuracy_change_from_k1",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
    "sequence_accuracy_change_from_k1",
)

CATEGORY_GENERALIZATION_FIELDS = (
    "architecture",
    "training_condition",
    "evaluated_unseen_task_count",
    "seed_count",
    "unseen_loss_mean",
    "unseen_loss_sample_sd",
    "seen_loss_mean",
    "loss_gap_unseen_minus_seen_mean",
    "loss_gap_sample_sd",
    "unseen_token_accuracy_mean",
    "unseen_token_accuracy_sample_sd",
    "seen_token_accuracy_mean",
    "token_accuracy_gap_unseen_minus_seen_mean",
    "token_accuracy_gap_sample_sd",
    "unseen_sequence_accuracy_mean",
    "unseen_sequence_accuracy_sample_sd",
    "seen_sequence_accuracy_mean",
    "sequence_accuracy_gap_unseen_minus_seen_mean",
    "sequence_accuracy_gap_sample_sd",
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


def rows_from_test_evaluations(
    config: Mapping[str, Any],
    evaluation_dir: Path,
) -> list[dict[str, Any]]:
    """Load and validate all 48 one-time test results into 960 rows."""

    manifest_path = evaluation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("status") != "completed"
        or manifest.get("run_count") != 48
        or manifest.get("examples_per_run") != 100_000
        or manifest.get("examples_per_task_per_run") != 5_000
    ):
        raise ValueError("test evaluation manifest is incomplete or invalid")
    task_names = tuple(task_names_for_experiment(dict(config)))
    holdouts = set(config["holdout_tasks"])
    category_groups = _category_groups(config)
    matched_category_tasks = {
        task for tasks in category_groups.values() for task in tasks
    }
    run_entries = manifest.get("runs")
    if not isinstance(run_entries, list) or len(run_entries) != 48:
        raise ValueError("test evaluation manifest run list is invalid")
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for entry in run_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("test evaluation manifest contains a malformed run")
        result_path = evaluation_dir / str(entry["result_file"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "completed"
            or result.get("run_id") != entry.get("run_id")
            or result.get("matrix") != entry.get("matrix")
            or result.get("checkpoint_sha256") != entry.get("checkpoint_sha256")
            or result.get("test_manifest_sha256") != manifest.get("test_manifest_sha256")
            or result.get("evaluator_commit") != manifest.get("evaluator_commit")
            or result.get("examples") != 100_000
        ):
            raise ValueError(f"test result identity is invalid: {result_path}")
        run_id = str(result["run_id"])
        if run_id in seen_run_ids:
            raise ValueError(f"duplicate test run: {run_id}")
        seen_run_ids.add(run_id)
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != set(task_names):
            raise ValueError(f"test result task grid is invalid: {run_id}")
        matrix = str(result["matrix"])
        trained_tasks = set(result["trained_tasks"])
        for task in task_names:
            metric = metrics[task]
            if metric.get("examples") != 5_000:
                raise ValueError(f"test count is invalid: {run_id}:{task}")
            row = {
                "protocol_version": config["protocol_version"],
                "matrix": matrix,
                "condition": result.get("condition", ""),
                "run_id": run_id,
                "architecture": result["architecture"],
                "trained_task_count": result["trained_task_count"],
                "seed": result["seed"],
                "task": task,
                "task_family": _task_family(task),
                "task_status": _task_status(
                    matrix=matrix,
                    task=task,
                    trained_tasks=trained_tasks,
                    holdouts=holdouts,
                    matched_category_tasks=matched_category_tasks,
                ),
                "evaluation_split": "test_shard_099",
                "examples": metric["examples"],
                "supervised_tokens": metric["tokens"],
                "loss": metric["loss"],
                "token_accuracy": metric["token_accuracy"],
                "sequence_accuracy": metric["sequence_accuracy"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "experiment_config_sha256": result["experiment_config_sha256"],
                "test_manifest_sha256": result["test_manifest_sha256"],
                "evaluation_shards": result["test_shards"],
                "evaluator_commit": result["evaluator_commit"],
            }
            for key in ("loss", "token_accuracy", "sequence_accuracy"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"non-finite {key} in test result {run_id}:{task}")
            rows.append(row)
    if len(rows) != 960:
        raise ValueError(f"expected 960 test rows, found {len(rows)}")
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
                    "task_macro_loss": _mean(
                        float(row["loss"]) for row in group_rows
                    ),
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
        loss = [float(row["task_macro_loss"]) for row in group]
        record = dict(zip(key_fields, key, strict=True))
        record.update(
            {
                "seed_count": 3,
                "loss_mean": statistics.fmean(loss),
                "loss_sample_sd": statistics.stdev(loss),
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


def build_nested_generalization(
    nested_summary: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Exclude seen tasks and report changes relative to each k=1 baseline."""

    eligible = [row for row in nested_summary if row["task_status"] != "seen"]
    baselines = {
        (row["architecture"], row["task_status"]): row
        for row in eligible
        if int(row["trained_task_count"]) == 1
    }
    output: list[dict[str, Any]] = []
    for row in eligible:
        status = str(row["task_status"])
        baseline = baselines[(row["architecture"], status)]
        trained_count = int(row["trained_task_count"])
        evaluated_count = 4 if status == "fixed_train_holdout" else 16 - trained_count
        output.append(
            {
                "architecture": row["architecture"],
                "trained_task_count": trained_count,
                "evaluation_group": status,
                "evaluated_unseen_task_count": evaluated_count,
                "seed_count": row["seed_count"],
                "loss_mean": row["loss_mean"],
                "loss_sample_sd": row["loss_sample_sd"],
                "loss_change_from_k1": float(row["loss_mean"])
                - float(baseline["loss_mean"]),
                "token_accuracy_mean": row["token_accuracy_mean"],
                "token_accuracy_sample_sd": row["token_accuracy_sample_sd"],
                "token_accuracy_change_from_k1": float(row["token_accuracy_mean"])
                - float(baseline["token_accuracy_mean"]),
                "sequence_accuracy_mean": row["sequence_accuracy_mean"],
                "sequence_accuracy_sample_sd": row["sequence_accuracy_sample_sd"],
                "sequence_accuracy_change_from_k1": float(row["sequence_accuracy_mean"])
                - float(baseline["sequence_accuracy_mean"]),
            }
        )
    return output


def build_category_generalization(
    run_summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Macro-average only off-diagonal matched-category tasks for each run."""

    category_rows = [row for row in run_summaries if row["matrix"] == "category"]
    by_run: dict[str, list[Mapping[str, Any]]] = {}
    for row in category_rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)
    per_seed: list[dict[str, Any]] = []
    for run_id, rows in by_run.items():
        first = rows[0]
        condition = first["condition"]
        seen = next(row for row in rows if row["evaluation_group"] == condition)
        unseen = [row for row in rows if row["evaluation_group"] != condition]
        if len(unseen) != 2 or any(int(row["task_count"]) != 4 for row in unseen):
            raise ValueError(f"category run {run_id} lacks two four-task unseen groups")
        record: dict[str, Any] = {
            "architecture": first["architecture"],
            "training_condition": condition,
            "seed": first["seed"],
            "evaluated_unseen_task_count": 8,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            seen_value = float(seen[f"task_macro_{metric}"])
            unseen_value = statistics.fmean(
                float(row[f"task_macro_{metric}"]) for row in unseen
            )
            record[f"seen_{metric}"] = seen_value
            record[f"unseen_{metric}"] = unseen_value
            record[f"{metric}_gap"] = unseen_value - seen_value
        per_seed.append(record)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in per_seed:
        grouped.setdefault(
            (str(row["architecture"]), str(row["training_condition"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    for (architecture, condition), rows in grouped.items():
        if len(rows) != 3 or len({row["seed"] for row in rows}) != 3:
            raise ValueError("category generalization group must contain three seeds")
        record = {
            "architecture": architecture,
            "training_condition": condition,
            "evaluated_unseen_task_count": 8,
            "seed_count": 3,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            unseen_values = [float(row[f"unseen_{metric}"]) for row in rows]
            seen_values = [float(row[f"seen_{metric}"]) for row in rows]
            gap_values = [float(row[f"{metric}_gap"]) for row in rows]
            record[f"unseen_{metric}_mean"] = statistics.fmean(unseen_values)
            record[f"unseen_{metric}_sample_sd"] = statistics.stdev(unseen_values)
            record[f"seen_{metric}_mean"] = statistics.fmean(seen_values)
            record[f"{metric}_gap_unseen_minus_seen_mean"] = statistics.fmean(
                gap_values
            )
            record[f"{metric}_gap_sample_sd"] = statistics.stdev(gap_values)
        output.append(record)
    return output


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


def export_results(
    config_path: Path,
    output_dir: Path,
    *,
    test_evaluation_dir: Path | None = None,
) -> dict[str, Any]:
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
        "validation": output_dir / "validation_model_task_accuracies.csv",
        "runs": output_dir / "validation_run_summaries.csv",
        "nested": output_dir / "validation_nested_summary.csv",
        "category": output_dir / "validation_category_summary.csv",
    }
    write_csv_atomic(outputs["validation"], validation_rows, VALIDATION_FIELDS)
    write_csv_atomic(outputs["runs"], run_summaries, RUN_SUMMARY_FIELDS)
    write_csv_atomic(outputs["nested"], nested_summary, NESTED_SUMMARY_FIELDS)
    write_csv_atomic(outputs["category"], category_summary, CATEGORY_SUMMARY_FIELDS)
    summary = {
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "validation_rows": len(validation_rows),
        "run_summary_rows": len(run_summaries),
        "nested_summary_rows": len(nested_summary),
        "category_summary_rows": len(category_summary),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    if test_evaluation_dir is not None:
        test_rows = rows_from_test_evaluations(config, test_evaluation_dir)
        test_run_summaries = build_run_summaries(test_rows, config)
        test_nested_summary = build_nested_summary(test_run_summaries)
        test_category_summary = build_category_summary(test_run_summaries)
        test_nested_generalization = build_nested_generalization(
            test_nested_summary
        )
        test_category_generalization = build_category_generalization(
            test_run_summaries
        )
        test_outputs = {
            "test": output_dir / "test_model_task_accuracies.csv",
            "test_runs": output_dir / "test_run_summaries.csv",
            "test_nested": output_dir / "test_nested_summary.csv",
            "test_category": output_dir / "test_category_summary.csv",
            "test_nested_generalization": (
                output_dir / "test_nested_generalization.csv"
            ),
            "test_category_generalization": (
                output_dir / "test_category_generalization.csv"
            ),
        }
        write_csv_atomic(test_outputs["test"], test_rows, TEST_FIELDS)
        write_csv_atomic(test_outputs["test_runs"], test_run_summaries, RUN_SUMMARY_FIELDS)
        write_csv_atomic(test_outputs["test_nested"], test_nested_summary, NESTED_SUMMARY_FIELDS)
        write_csv_atomic(test_outputs["test_category"], test_category_summary, CATEGORY_SUMMARY_FIELDS)
        write_csv_atomic(
            test_outputs["test_nested_generalization"],
            test_nested_generalization,
            NESTED_GENERALIZATION_FIELDS,
        )
        write_csv_atomic(
            test_outputs["test_category_generalization"],
            test_category_generalization,
            CATEGORY_GENERALIZATION_FIELDS,
        )
        summary.update(
            {
                "test_rows": len(test_rows),
                "test_run_summary_rows": len(test_run_summaries),
                "test_nested_summary_rows": len(test_nested_summary),
                "test_category_summary_rows": len(test_category_summary),
                "test_nested_generalization_rows": len(
                    test_nested_generalization
                ),
                "test_category_generalization_rows": len(
                    test_category_generalization
                ),
            }
        )
        summary["outputs"].update(
            {key: str(value) for key, value in test_outputs.items()}
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-evaluation-dir",
        type=Path,
        default=None,
        help="also export one-time test metrics from this completed evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_results(
        args.config,
        args.output_dir,
        test_evaluation_dir=args.test_evaluation_dir,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

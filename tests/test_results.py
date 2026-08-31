from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib

from neurips_permutations.experiments import build_experiment_matrix
from neurips_permutations.results import (
    CATEGORY_GENERALIZATION_FIELDS,
    CATEGORY_SUMMARY_FIELDS,
    NESTED_GENERALIZATION_FIELDS,
    NESTED_SUMMARY_FIELDS,
    RUN_SUMMARY_FIELDS,
    TEST_FIELDS,
    VALIDATION_FIELDS,
    build_category_generalization,
    build_category_summary,
    build_nested_generalization,
    build_nested_summary,
    build_run_summaries,
    validation_rows_from_audits,
    rows_from_test_evaluations,
    write_csv_atomic,
)


CONFIG = Path(__file__).parents[1] / "configs" / "henry_permutation_revised.toml"


def _fake_audit(matrix: str) -> dict:
    runs = []
    for run in build_experiment_matrix(CONFIG, matrix=matrix):
        validation = {
            task: {
                "examples": 10,
                "tokens": 20,
                "loss": float(index + 1),
                "token_accuracy": (index + 1) / 100,
                "sequence_accuracy": index / 100,
            }
            for index, task in enumerate(
                tomllib.loads(CONFIG.read_text(encoding="utf-8"))["task_order"]
            )
        }
        record = {
            "status": "passed",
            "run_id": run.run_id,
            "architecture": run.architecture,
            "task_count": run.task_count,
            "tasks": list(run.tasks),
            "seed": run.seed,
            "checkpoint_sha256": "c" * 64,
            "results": {"validation": validation},
        }
        if hasattr(run, "condition"):
            record["condition"] = run.condition
        runs.append(record)
    return {
        "ok": True,
        "config_sha256": "e" * 64,
        "manifests": {"validation": {"sha256": "m" * 64}},
        "runs": runs,
    }


def _rows_and_config() -> tuple[list[dict], dict]:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    rows = validation_rows_from_audits(
        config,
        {"nested": _fake_audit("nested"), "category": _fake_audit("category")},
        validation_split_manifest_sha256="s" * 64,
    )
    return rows, config


def test_validation_export_contains_every_model_task_cell_and_status() -> None:
    rows, _ = _rows_and_config()
    assert len(rows) == 960
    assert sum(row["matrix"] == "nested" for row in rows) == 600
    assert sum(row["matrix"] == "category" for row in rows) == 360
    assert len({(row["matrix"], row["run_id"], row["task"]) for row in rows}) == 960

    nested_statuses = [row["task_status"] for row in rows if row["matrix"] == "nested"]
    assert nested_statuses.count("seen") == 186
    assert nested_statuses.count("pool_unseen") == 294
    assert nested_statuses.count("fixed_train_holdout") == 120

    category_statuses = [row["task_status"] for row in rows if row["matrix"] == "category"]
    assert category_statuses.count("seen") == 72
    assert category_statuses.count("cross_category_matched") == 144
    assert category_statuses.count("other_unmatched_statistics") == 144
    assert {row["evaluation_shards"] for row in rows} == {"098"}


def test_task_macro_summaries_average_within_run_then_across_seeds() -> None:
    rows, config = _rows_and_config()
    runs = build_run_summaries(rows, config)
    nested = build_nested_summary(runs)
    category = build_category_summary(runs)

    assert len(runs) == 138
    assert len(nested) == 28
    assert len(category) == 18
    assert {row["seed_count"] for row in nested + category} == {3}
    assert all(row["loss_sample_sd"] == 0 for row in nested + category)
    assert all(row["token_accuracy_sample_sd"] == 0 for row in nested + category)
    assert all(row["sequence_accuracy_sample_sd"] == 0 for row in nested + category)

    encoding_to_encoding = next(
        row
        for row in category
        if row["architecture"] == "transformer"
        and row["training_condition"] == "encoding_e4"
        and row["evaluation_condition"] == "encoding_e4"
    )
    # The four encoding tasks occupy one-based task-order positions 9, 20, 4, and 17.
    assert encoding_to_encoding["token_accuracy_mean"] == (0.09 + 0.20 + 0.04 + 0.17) / 4


def test_csv_writer_preserves_declared_schema_and_replaces_atomically(tmp_path: Path) -> None:
    rows, config = _rows_and_config()
    run_rows = build_run_summaries(rows, config)
    nested_rows = build_nested_summary(run_rows)
    category_rows = build_category_summary(run_rows)
    nested_generalization = build_nested_generalization(nested_rows)
    category_generalization = build_category_generalization(run_rows)
    fixtures = (
        ("raw.csv", rows[:2], VALIDATION_FIELDS),
        ("runs.csv", run_rows[:2], RUN_SUMMARY_FIELDS),
        ("nested.csv", nested_rows[:2], NESTED_SUMMARY_FIELDS),
        ("category.csv", category_rows[:2], CATEGORY_SUMMARY_FIELDS),
        (
            "nested-generalization.csv",
            nested_generalization[:2],
            NESTED_GENERALIZATION_FIELDS,
        ),
        (
            "category-generalization.csv",
            category_generalization[:2],
            CATEGORY_GENERALIZATION_FIELDS,
        ),
    )
    for name, records, fields in fixtures:
        path = tmp_path / name
        write_csv_atomic(path, records, fields)
        with path.open(newline="", encoding="utf-8") as handle:
            parsed = list(csv.DictReader(handle))
        assert parsed
        assert tuple(parsed[0]) == tuple(fields)
    assert not list(tmp_path.glob("*.tmp"))
    assert VALIDATION_FIELDS[0] == TEST_FIELDS[0] == "matrix"
    assert "protocol_version" not in VALIDATION_FIELDS
    assert "protocol_version" not in TEST_FIELDS


def test_generalization_summaries_exclude_seen_tasks() -> None:
    rows, config = _rows_and_config()
    run_rows = build_run_summaries(rows, config)
    nested = build_nested_generalization(build_nested_summary(run_rows))
    category = build_category_generalization(run_rows)

    assert len(nested) == 18
    assert {row["evaluation_group"] for row in nested} == {
        "pool_unseen",
        "fixed_train_holdout",
    }
    assert all(
        row["evaluated_unseen_task_count"] == 4
        for row in nested
        if row["evaluation_group"] == "fixed_train_holdout"
    )
    assert all(
        row["loss_change_from_k1"] == 0
        and row["token_accuracy_change_from_k1"] == 0
        and row["sequence_accuracy_change_from_k1"] == 0
        for row in nested
        if row["trained_task_count"] == 1
    )
    assert len(category) == 6
    assert {row["evaluated_unseen_task_count"] for row in category} == {8}


def test_failed_audit_is_rejected() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    nested = _fake_audit("nested")
    nested["ok"] = False
    try:
        validation_rows_from_audits(
            config,
            {"nested": nested, "category": _fake_audit("category")},
            validation_split_manifest_sha256="s" * 64,
        )
    except ValueError as error:
        assert "nested strict audit did not pass" in str(error)
    else:
        raise AssertionError("failed audit was accepted")


def test_completed_test_evaluation_exports_all_960_cells(tmp_path: Path) -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    task_names = tuple(config["task_order"])
    entries = []
    for matrix in ("nested", "category"):
        for run in build_experiment_matrix(CONFIG, matrix=matrix):
            relative = Path("per-run") / f"{run.run_id}.json"
            result = {
                "status": "completed",
                "matrix": matrix,
                "condition": getattr(run, "condition", ""),
                "run_id": run.run_id,
                "architecture": run.architecture,
                "trained_tasks": list(run.tasks),
                "trained_task_count": run.task_count,
                "seed": run.seed,
                "checkpoint_sha256": "c" * 64,
                "experiment_config_sha256": "e" * 64,
                "test_manifest_sha256": "t" * 64,
                "test_shards": "099",
                "evaluator_commit": "g" * 40,
                "examples": 100_000,
                "metrics": {
                    task: {
                        "examples": 5_000,
                        "tokens": 10_000,
                        "loss": 1.0,
                        "token_accuracy": 0.5,
                        "sequence_accuracy": 0.25,
                    }
                    for task in task_names
                },
            }
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result), encoding="utf-8")
            entries.append(
                {
                    "matrix": matrix,
                    "run_id": run.run_id,
                    "checkpoint_sha256": "c" * 64,
                    "result_file": str(relative),
                }
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_count": 48,
                "examples_per_run": 100_000,
                "examples_per_task_per_run": 5_000,
                "test_manifest_sha256": "t" * 64,
                "evaluator_commit": "g" * 40,
                "runs": entries,
            }
        ),
        encoding="utf-8",
    )
    rows = rows_from_test_evaluations(config, tmp_path)
    assert len(rows) == 960
    assert {row["evaluation_split"] for row in rows} == {"test_shard_099"}
    assert {row["examples"] for row in rows} == {5_000}
    assert set(TEST_FIELDS) < set(rows[0])

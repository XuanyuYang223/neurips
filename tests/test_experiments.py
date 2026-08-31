from __future__ import annotations

from pathlib import Path

from neurips_permutations.experiments import (
    _completion_is_valid,
    _read_config,
    _training_command,
    build_matrix,
    matrix_summary,
    task_names_for_experiment,
)
from neurips_permutations.math_ops import V2_TASK_NAMES, V3_TASK_NAMES
from neurips_permutations.training import build_arg_parser


CONFIG = Path(__file__).parents[1] / "configs" / "henry_permutation.toml"
REVISED_CONFIG = (
    Path(__file__).parents[1] / "configs" / "henry_permutation_revised.toml"
)


def test_frozen_matrix_contains_thirty_unique_nested_runs() -> None:
    runs = build_matrix(CONFIG)
    assert len(runs) == 30
    assert len({run.run_id for run in runs}) == 30
    assert {run.architecture for run in runs} == {"transformer", "mlp"}
    assert {run.task_count for run in runs} == {1, 2, 4, 8, 16}
    assert len({run.seed for run in runs}) == 3

    by_size = {}
    for run in runs:
        by_size.setdefault(run.task_count, run.tasks)
        assert by_size[run.task_count] == run.tasks
    assert by_size[1] == by_size[2][:1]
    assert by_size[2] == by_size[4][:2]
    assert by_size[4] == by_size[8][:4]
    assert by_size[8] == by_size[16][:8]


def test_revised_v3_matrix_parses_as_thirty_nested_runs() -> None:
    config, _ = _read_config(REVISED_CONFIG)
    runs = build_matrix(REVISED_CONFIG)

    assert task_names_for_experiment(config) == V3_TASK_NAMES
    assert len(runs) == 30
    assert len({run.run_id for run in runs}) == 30
    assert {run.task_count for run in runs} == {1, 2, 4, 8, 16}
    assert all(set(run.tasks) <= set(V3_TASK_NAMES) for run in runs)
    assert any(
        {"peaks", "exceedances", "recoils"} <= set(run.tasks)
        for run in runs
    )
    assert all(
        {"power", "conjugate", "commutator"}.isdisjoint(run.tasks)
        for run in runs
    )
    # Declarative category-comparison metadata must not add runs to this
    # existing nested matrix.
    assert "category_comparison" in config


def test_legacy_experiment_without_dataset_protocol_defaults_to_v2() -> None:
    config, _ = _read_config(CONFIG)
    assert "dataset_protocol_version" not in config
    assert task_names_for_experiment(config) == V2_TASK_NAMES


def test_status_starts_with_every_run_incomplete(tmp_path: Path) -> None:
    config = tmp_path / "henry_permutation.toml"
    config.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            'output_dir = "runs/henry-permutation"',
            f'output_dir = "{tmp_path / "runs"}"',
        ),
        encoding="utf-8",
    )

    summary = matrix_summary(config)
    assert summary["run_count"] == 30
    assert summary["complete_count"] == 0
    assert summary["incomplete_count"] == 30


def test_completion_marker_requires_matching_config_and_checkpoint(tmp_path: Path) -> None:
    import hashlib
    import json

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker = tmp_path / "completed.json"
    value = {
        "status": "completed",
        "global_step": 20,
        "experiment_config_sha256": "config-hash",
        "config_sha256": "run-hash",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
    }
    marker.write_text(json.dumps(value), encoding="utf-8")
    assert _completion_is_valid(
        marker, experiment_config_sha256="config-hash", expected_steps=20
    )
    checkpoint.write_bytes(b"corrupt")
    assert not _completion_is_valid(
        marker, experiment_config_sha256="config-hash", expected_steps=20
    )
    marker.write_text("garbage", encoding="utf-8")
    assert not _completion_is_valid(
        marker, experiment_config_sha256="config-hash", expected_steps=20
    )


def test_training_command_carries_every_frozen_launch_parameter() -> None:
    config, _ = _read_config(CONFIG)
    run = build_matrix(CONFIG)[0]
    command = _training_command(run, config, CONFIG)
    args = build_arg_parser().parse_args(command[3:])
    assert args.manifest == config["dataset_manifest"]
    assert args.validation_manifest == config["validation_manifest"]
    assert args.train_shard_indices == config["data"]["train_shards"]
    assert args.validation_shard_indices == config["data"]["validation_shards"]
    assert args.dropout == config["model"]["dropout"]
    assert args.warmup_steps == config["training"]["warmup_steps"]
    assert (
        args.validation_batches_per_task
        == config["training"]["validation_batches_per_task"]
    )
    assert args.max_tokens_per_batch == config["training"]["max_tokens_per_batch"]

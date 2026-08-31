from __future__ import annotations

from pathlib import Path

import pytest

from neurips_permutations.experiments import (
    _completion_is_valid,
    _read_config,
    _training_command,
    build_category_matrix,
    build_experiment_matrix,
    build_matrix,
    matrix_summary,
    run_matrix,
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


def test_revised_category_matrix_contains_eighteen_matched_runs() -> None:
    runs = build_category_matrix(REVISED_CONFIG)

    assert len(runs) == 18
    assert len({run.run_id for run in runs}) == 18
    assert {run.architecture for run in runs} == {"transformer", "mlp"}
    assert {run.seed for run in runs} == {17, 42, 314159}
    assert {run.task_count for run in runs} == {4}
    expected = {
        "encoding_e4": (
            "to_cycle",
            "to_lehmer",
            "to_inversion_vector",
            "to_reduced_word",
        ),
        "statistics_s4": (
            "length",
            "cycle_type",
            "rsk_shape",
            "pattern_avoidance",
        ),
        "algebra_a4": (
            "inverse",
            "compose",
            "right_multiply_simple",
            "bruhat_leq",
        ),
    }
    expected_batches = {
        "encoding_e4": (4, 16),
        "statistics_s4": (16, 4),
        "algebra_a4": (16, 4),
    }
    for condition, tasks in expected.items():
        condition_runs = [run for run in runs if run.condition == condition]
        assert len(condition_runs) == 6
        assert all(run.tasks == tasks for run in condition_runs)
        assert all("/category-comparison/" in run.output_dir for run in condition_runs)
        micro_batch_size, gradient_accumulation_steps = expected_batches[condition]
        assert all(
            run.micro_batch_size == micro_batch_size for run in condition_runs
        )
        assert all(
            run.gradient_accumulation_steps == gradient_accumulation_steps
            for run in condition_runs
        )
        assert micro_batch_size * gradient_accumulation_steps == 64


def test_matrix_dispatch_preserves_nested_default_and_selects_category() -> None:
    assert build_experiment_matrix(REVISED_CONFIG) == build_matrix(REVISED_CONFIG)
    assert build_experiment_matrix(
        REVISED_CONFIG, matrix="category"
    ) == build_category_matrix(REVISED_CONFIG)
    with pytest.raises(ValueError, match="unknown experiment matrix"):
        build_experiment_matrix(REVISED_CONFIG, matrix="other")  # type: ignore[arg-type]


def test_category_matrix_rejects_mutated_frozen_task_set(tmp_path: Path) -> None:
    config = tmp_path / "revised.toml"
    text = REVISED_CONFIG.read_text(encoding="utf-8").replace(
        'tasks = ["length", "cycle_type", "rsk_shape", "pattern_avoidance"]',
        'tasks = ["length", "cycle_type", "rsk_shape", "peaks"]',
    )
    config.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="frozen four-task set"):
        build_category_matrix(config)


def test_category_matrix_rejects_unfair_effective_batch(tmp_path: Path) -> None:
    config = tmp_path / "revised.toml"
    text = REVISED_CONFIG.read_text(encoding="utf-8").replace(
        'name = "encoding_e4"\n'
        'tasks = ["to_cycle", "to_lehmer", "to_inversion_vector", "to_reduced_word"]\n'
        "micro_batch_size = 4\n"
        "gradient_accumulation_steps = 16",
        'name = "encoding_e4"\n'
        'tasks = ["to_cycle", "to_lehmer", "to_inversion_vector", "to_reduced_word"]\n'
        "micro_batch_size = 4\n"
        "gradient_accumulation_steps = 8",
    )
    config.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="effective batch size 64"):
        build_category_matrix(config)


def test_legacy_v2_config_has_no_category_matrix() -> None:
    with pytest.raises(ValueError, match="category_comparison"):
        build_category_matrix(CONFIG)


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
    assert args.batch_size == config["training"]["micro_batch_size"] == 16
    assert args.grad_accum == config["training"]["gradient_accumulation_steps"] == 4


def test_category_training_command_uses_existing_resumable_entry_point() -> None:
    config, _ = _read_config(REVISED_CONFIG)
    run = build_category_matrix(REVISED_CONFIG)[0]
    command = _training_command(run, config, REVISED_CONFIG)
    args = build_arg_parser().parse_args(command[3:])

    assert command[1:3] == ["-m", "neurips_permutations.training"]
    assert args.output_dir == run.output_dir
    assert tuple(args.tasks[0].split(",")) == run.tasks
    assert args.architecture == run.architecture
    assert args.seed == run.seed
    assert args.batch_size == run.micro_batch_size == 4
    assert args.grad_accum == run.gradient_accumulation_steps == 16
    assert args.batch_size * args.grad_accum == 64
    assert args.resume == "auto"
    assert args.experiment_config == str(REVISED_CONFIG)


def test_category_dry_run_emits_all_commands_without_starting_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json
    import neurips_permutations.experiments as experiments

    monkeypatch.setattr(experiments, "_sha256_file", lambda _: "0" * 64)

    def fail_subprocess(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not start a training subprocess")

    monkeypatch.setattr(experiments.subprocess, "run", fail_subprocess)
    assert run_matrix(REVISED_CONFIG, matrix="category", dry_run=True) == 0

    plans = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(plans) == 18
    assert len({plan["run_id"] for plan in plans}) == 18
    assert all(
        plan["command"][1:3] == ["-m", "neurips_permutations.training"]
        for plan in plans
    )
    expected_batches = {
        "encoding-e4": (4, 16),
        "statistics-s4": (16, 4),
        "algebra-a4": (16, 4),
    }
    for plan in plans:
        args = build_arg_parser().parse_args(plan["command"][3:])
        condition = next(
            name for name in expected_batches if f"category-{name}-" in plan["run_id"]
        )
        assert (args.batch_size, args.grad_accum) == expected_batches[condition]
        assert args.batch_size * args.grad_accum == 64

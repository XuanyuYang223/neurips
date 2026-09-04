from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import tomllib
from typing import Any, Callable

import pytest
import torch
from torch.optim import AdamW

from neurips_permutations.audit import (
    AuditPathError,
    audit_experiment,
    expected_train_config,
    main,
    training_config_sha256,
)
from neurips_permutations.experiments import (
    ExperimentRun,
    build_category_matrix,
    build_matrix,
    task_names_for_experiment,
)
from neurips_permutations.math_ops import TASK_NAMES, V3_TASK_NAMES
from neurips_permutations.models import build_model
from neurips_permutations.passage import TOKEN_TO_ID
from neurips_permutations.training import TrainConfig, _model_vocab_size, _scheduler


SOURCE_CONFIG = Path(__file__).parents[1] / "configs" / "henry_permutation.toml"
REVISED_SOURCE_CONFIG = (
    Path(__file__).parents[1] / "configs" / "henry_permutation_revised.toml"
)


@dataclass(frozen=True)
class AuditFixture:
    repository: Path
    config_path: Path
    output_root: Path
    manifest_path: Path
    manifest_sha256: str


def _replace_toml(
    text: str,
    key: str,
    value: Any,
    *,
    first_only: bool = False,
) -> str:
    rendered = json.dumps(value) if isinstance(value, (str, bool)) else str(value)
    rendered = rendered.lower() if isinstance(value, bool) else rendered
    updated, count = re.subn(
        rf"^{re.escape(key)}\s*=.*$",
        f"{key} = {rendered}",
        text,
        count=1 if first_only else 0,
        flags=re.M,
    )
    assert count == 1, key
    return updated


def _manifest(
    indices: tuple[int, ...],
    *,
    parent_sha256: str | None = None,
    task_names: tuple[str, ...] = TASK_NAMES,
    schema_version: str = "permutation-20/v2",
) -> dict[str, Any]:
    per_shard = 100_000 // len(task_names)
    shards = [
        {
            "byte_size": 1,
            "filename": f"part-{index:05d}.jsonl.gz",
            "first_id": index * 100_000,
            "index": index,
            "last_id": (index + 1) * 100_000 - 1,
            "record_count": 100_000,
            "sha256": hashlib.sha256(b"x").hexdigest(),
            "task_counts": {task: per_shard for task in task_names},
        }
        for index in indices
    ]
    value: dict[str, Any] = {
        "base": 100,
        "count": len(indices) * 100_000,
        "format": "jsonl.gz",
        "gzip_compresslevel": 6,
        "gzip_mtime": 0,
        "max_entries": 30,
        "schema_version": schema_version,
        "seed": 20260830,
        "shard_count": len(indices),
        "shard_size": 100_000,
        "shards": shards,
        "task_counts": {task: per_shard * len(indices) for task in task_names},
        "tasks": list(task_names),
        "total_bytes": len(indices),
    }
    if parent_sha256 is not None:
        value.update(
            {
                "parent_manifest": "manifest.json",
                "parent_manifest_sha256": parent_sha256,
                "split": "test",
            }
        )
    return value


def _make_fixture(tmp_path: Path, *, revised: bool = False) -> AuditFixture:
    repository = tmp_path / "repo"
    config_dir = repository / "configs"
    data_name = "permutation-10m-v3" if revised else "permutation-10m-v2"
    schema_version = "permutation-20/v3" if revised else "permutation-20/v2"
    task_names = V3_TASK_NAMES if revised else TASK_NAMES
    output_name = "henry-permutation-v3" if revised else "henry-permutation"
    data_dir = repository / "data" / data_name
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    for index in range(100):
        (data_dir / f"part-{index:05d}.jsonl.gz").write_bytes(b"x")

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                tuple(range(100)),
                task_names=task_names,
                schema_version=schema_version,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (data_dir / "test_manifest.json").write_text(
        json.dumps(
            _manifest(
                (99,),
                parent_sha256=manifest_sha256,
                task_names=task_names,
                schema_version=schema_version,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    source_config = REVISED_SOURCE_CONFIG if revised else SOURCE_CONFIG
    text = source_config.read_text(encoding="utf-8")
    replacements = {
        "dataset_manifest": f"data/{data_name}/manifest.json",
        "validation_manifest": f"data/{data_name}/manifest.json",
        "test_manifest": f"data/{data_name}/test_manifest.json",
        "output_dir": f"runs/{output_name}",
        "max_sequence_length": 16,
        "shuffle_buffer": 10,
        "d_model": 8,
        "transformer_layers": 1,
        "mlp_layers": 1,
        "num_heads": 2,
        "ff_multiplier": 2,
        "max_steps": 3,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_tokens_per_batch": 32,
        "warmup_steps": 1,
        "checkpoint_every_steps": 1,
        "validate_every_steps": 1,
        "validation_batches_per_task": 1,
    }
    for key, value in replacements.items():
        text = _replace_toml(
            text,
            key,
            value,
            first_only=key in {"micro_batch_size", "gradient_accumulation_steps"},
        )
    config_path = config_dir / "experiment.toml"
    config_path.write_text(text, encoding="utf-8")
    return AuditFixture(
        repository=repository,
        config_path=config_path,
        output_root=repository / "runs" / output_name,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "model_type": config["architecture"],
        "vocab_size": _model_vocab_size(TrainConfig.from_value(config)),
        "max_seq_len": config["max_seq_len"],
        "d_model": config["d_model"],
        "layers": config["num_layers"],
        "dropout": config["dropout"],
        "mlp_ratio": config["mlp_ratio"],
        "tie_embeddings": config["tie_embeddings"],
    }
    if config["architecture"] == "transformer":
        kwargs["n_heads"] = config["num_heads"]
    return kwargs


def _validation(
    task_names: tuple[str, ...] = TASK_NAMES,
) -> dict[str, dict[str, float | int]]:
    return {
        task: {
            "loss": 0.75,
            "token_accuracy": 0.5,
            "sequence_accuracy": 0.25,
            "tokens": 20,
            "examples": 10,
        }
        for task in task_names
    }


def _write_completed_run(
    fixture: AuditFixture,
    run: ExperimentRun | None = None,
) -> tuple[ExperimentRun, Path, Path]:
    experiment = tomllib.loads(fixture.config_path.read_text(encoding="utf-8"))
    run = run or build_matrix(fixture.config_path)[0]
    config_sha256 = hashlib.sha256(fixture.config_path.read_bytes()).hexdigest()
    config = expected_train_config(
        run,
        experiment,
        fixture.config_path,
        experiment_config_sha256=config_sha256,
    )
    model = build_model(**_model_kwargs(config))
    train_config = TrainConfig.from_value(config)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = _scheduler(optimizer, train_config)
    for _ in range(train_config.max_steps):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    examples = {task: 10 for task in run.tasks}
    tokens = {task: 20 for task in run.tasks}
    loss_sums = {task: 5.0 for task in run.tasks}
    validation = _validation(task_names_for_experiment(experiment))
    state = {
        "epoch": 2,
        "batches_in_epoch": 7,
        "global_step": train_config.max_steps,
        "task_examples": examples,
        "task_supervised_tokens": tokens,
        "task_loss_sum": loss_sums,
    }
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": {},
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": [torch.arange(16, dtype=torch.uint8)],
        },
        "state": state,
        "config": config,
        "data_fingerprints": {
            "training_manifest_sha256": fixture.manifest_sha256,
            "validation_manifest_sha256": fixture.manifest_sha256,
        },
        "validation": validation,
    }
    run_dir = fixture.repository / run.output_dir
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    marker = {
        "status": "completed",
        "run_id": run.run_id,
        "architecture": run.architecture,
        "tasks": list(run.tasks),
        "seed": run.seed,
        "global_step": train_config.max_steps,
        "epoch": state["epoch"],
        "batches_in_epoch": state["batches_in_epoch"],
        "last_loss": 0.5,
        "checkpoint": f"{run.output_dir}/checkpoint.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "config_sha256": training_config_sha256(config),
        "experiment_config_sha256": config_sha256,
        "training_manifest_sha256": fixture.manifest_sha256,
        "validation_manifest_sha256": fixture.manifest_sha256,
        "task_accounting": {
            task: {
                "examples": examples[task],
                "supervised_tokens": tokens[task],
                "mean_example_loss": loss_sums[task] / examples[task],
            }
            for task in run.tasks
        },
        "validation": validation,
    }
    marker_path = run_dir / "completed.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    return run, checkpoint_path, marker_path


def _rewrite_checkpoint(
    checkpoint_path: Path,
    marker_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    *,
    refresh_config_digest: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mutate(checkpoint)
    torch.save(checkpoint, checkpoint_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if refresh_config_digest:
        marker["config_sha256"] = training_config_sha256(checkpoint["config"])
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def _run_result(summary: dict[str, Any], run: ExperimentRun) -> dict[str, Any]:
    return next(value for value in summary["runs"] if value["run_id"] == run.run_id)


def _codes(result: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in result["issues"]}


def test_valid_formal_checkpoint_passes_while_other_runs_are_incomplete(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, _ = _write_completed_run(fixture)

    summary = audit_experiment(fixture.config_path)
    result = _run_result(summary, run)

    assert result["status"] == "passed"
    assert result["checkpoint_numeric"]["status"] == "passed"
    assert result["model_tensors"]["tensor_count"] > 0
    assert result["optimizer_tensors"]["tensor_count"] > 0
    assert summary["passed_count"] == 1
    assert summary["incomplete_count"] == 29
    assert summary["failed_count"] == 0
    assert all(value["status"] == "passed" for value in summary["manifests"].values())


def test_repository_contained_absolute_experiment_path_is_canonicalized(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)

    def use_absolute_path(value: dict[str, Any]) -> None:
        value["config"]["experiment_config"] = str(fixture.config_path)

    _rewrite_checkpoint(
        checkpoint,
        marker,
        use_absolute_path,
        refresh_config_digest=True,
    )
    result = _run_result(audit_experiment(fixture.config_path), run)

    assert result["status"] == "passed"
    assert not result["issues"]


def test_absolute_experiment_path_outside_repository_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    outside = tmp_path / "outside.toml"
    outside.write_bytes(fixture.config_path.read_bytes())
    run, checkpoint, marker = _write_completed_run(fixture)

    def escape_repository(value: dict[str, Any]) -> None:
        value["config"]["experiment_config"] = str(outside)

    _rewrite_checkpoint(
        checkpoint,
        marker,
        escape_repository,
        refresh_config_digest=True,
    )
    result = _run_result(audit_experiment(fixture.config_path), run)

    assert result["status"] == "failed"
    assert "checkpoint_config_path_invalid" in _codes(result)


def test_revised_v3_audit_accepts_schema_aware_manifests_and_plan(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path, revised=True)
    completed_run, _, _ = _write_completed_run(fixture)

    summary = audit_experiment(fixture.config_path)

    assert summary["run_count"] == 30
    assert summary["incomplete_count"] == 29
    assert summary["passed_count"] == 1
    assert summary["failed_count"] == 0
    assert _run_result(summary, completed_run)["status"] == "passed"
    assert all(value["status"] == "passed" for value in summary["manifests"].values())
    assert not summary["issues"]
    assert all(set(run["tasks"]) <= set(V3_TASK_NAMES) for run in summary["runs"])
    assert all(
        {"power", "conjugate", "commutator"}.isdisjoint(run["tasks"])
        for run in summary["runs"]
    )


def test_category_checkpoint_uses_the_same_strict_eighteen_run_audit(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path, revised=True)
    category_run = build_category_matrix(fixture.config_path)[0]
    experiment = tomllib.loads(fixture.config_path.read_text(encoding="utf-8"))
    expected_config = expected_train_config(
        category_run,
        experiment,
        fixture.config_path,
    )
    assert expected_config["batch_size"] == 4
    assert expected_config["gradient_accumulation_steps"] == 16
    assert (
        expected_config["batch_size"]
        * expected_config["gradient_accumulation_steps"]
        == 64
    )
    completed_run, _, _ = _write_completed_run(fixture, category_run)

    summary = audit_experiment(fixture.config_path, matrix="category")
    result = _run_result(summary, completed_run)

    assert summary["matrix"] == "category"
    assert summary["expected_run_count"] == 18
    assert summary["run_count"] == 18
    assert summary["passed_count"] == 1
    assert summary["incomplete_count"] == 17
    assert summary["failed_count"] == 0
    assert summary["output_root"].endswith("/category-comparison")
    assert result["status"] == "passed"
    assert result["condition"] == "encoding_e4"
    assert result["checkpoint_numeric"]["status"] == "passed"
    assert all(value["status"] == "passed" for value in summary["manifests"].values())


def test_unmarked_checkpoint_is_never_deserialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    run = build_matrix(fixture.config_path)[0]
    run_dir = fixture.repository / run.output_dir
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"active checkpoint")

    def fail_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unmarked checkpoint must not be loaded")

    monkeypatch.setattr(torch, "load", fail_load)
    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "incomplete"
    assert _codes(result) == {"completion_marker_missing"}


def test_junk_model_tensor_is_rejected_by_expected_strict_model_load(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)
    _rewrite_checkpoint(
        checkpoint,
        marker,
        lambda value: value["model"].__setitem__("junk", torch.ones(1)),
    )

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert {"model_state_schema_mismatch", "model_strict_load_failed"} <= _codes(result)


def test_empty_model_state_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)
    _rewrite_checkpoint(checkpoint, marker, lambda value: value["model"].clear())

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "checkpoint_model_empty" in _codes(result)


def test_self_consistent_but_wrong_complete_train_config_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)

    def corrupt(value: dict[str, Any]) -> None:
        value["config"]["weight_decay"] = 0.5
        value["config"]["shard_indices"] = tuple(range(97))
        value["config"]["validation_shard_indices"] = (99,)
        value["config"]["d_model"] = 9
        value["config"]["amp"] = False

    _rewrite_checkpoint(checkpoint, marker, corrupt, refresh_config_digest=True)
    result = _run_result(audit_experiment(fixture.config_path), run)

    assert result["status"] == "failed"
    assert "checkpoint_full_config_mismatch" in _codes(result)
    assert "checkpoint_config_sha256_mismatch" in _codes(result)


@pytest.mark.parametrize("missing", ["scheduler", "rng", "validation"])
def test_missing_required_checkpoint_schema_is_rejected(
    tmp_path: Path, missing: str
) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)
    _rewrite_checkpoint(checkpoint, marker, lambda value: value.pop(missing))

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "checkpoint_schema_mismatch" in _codes(result)


def test_nested_checkpoint_nan_is_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)

    def corrupt(value: dict[str, Any]) -> None:
        value["scheduler"]["_last_lr"][0] = float("nan")

    _rewrite_checkpoint(checkpoint, marker, corrupt)
    result = _run_result(audit_experiment(fixture.config_path), run)

    assert result["status"] == "failed"
    assert "checkpoint_numbers_not_finite" in _codes(result)
    assert result["checkpoint_numeric"]["nonfinite"][0]["path"].endswith(
        "scheduler._last_lr[0]"
    )


def test_negative_losses_and_second_moment_are_rejected(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker_path = _write_completed_run(fixture)

    def corrupt(value: dict[str, Any]) -> None:
        task = run.tasks[0]
        value["state"]["task_loss_sum"][task] = -5.0
        first_state = next(iter(value["optimizer"]["state"].values()))
        first_state["exp_avg_sq"].view(-1)[0] = -1.0

    _rewrite_checkpoint(checkpoint, marker_path, corrupt)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["last_loss"] = -0.5
    marker["task_accounting"][run.tasks[0]]["mean_example_loss"] = -0.5
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    result = _run_result(audit_experiment(fixture.config_path), run)

    assert {
        "marker_last_loss_invalid",
        "state_task_loss_invalid",
        "optimizer_second_moment_negative",
    } <= _codes(result)


def test_supported_older_torch_optional_state_keys_are_not_required(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, marker = _write_completed_run(fixture)

    def older_schema(value: dict[str, Any]) -> None:
        value["optimizer"]["param_groups"][0].pop("decoupled_weight_decay", None)
        value["scheduler"].pop("_is_initial", None)

    _rewrite_checkpoint(checkpoint, marker, older_schema)
    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "passed"


@pytest.mark.parametrize("missing", ["task_accounting", "validation"])
def test_missing_marker_statistics_are_rejected(
    tmp_path: Path, missing: str
) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, marker_path = _write_completed_run(fixture)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop(missing)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "completion_marker_schema_mismatch" in _codes(result)
    assert f"marker_{missing}_missing" in _codes(result)


def test_marker_accounting_must_match_checkpoint_state(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, marker_path = _write_completed_run(fixture)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["task_accounting"][run.tasks[0]]["examples"] += 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "marker_task_examples_mismatch" in _codes(result)


def test_marker_validation_must_match_authenticated_checkpoint(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, marker_path = _write_completed_run(fixture)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["validation"][TASK_NAMES[0]]["loss"] += 1.0
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "marker_validation_checkpoint_mismatch" in _codes(result)


def test_checkpoint_symlink_is_rejected_and_never_followed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, checkpoint, _ = _write_completed_run(fixture)
    outside = fixture.repository / "outside.pt"
    checkpoint.rename(outside)
    checkpoint.symlink_to(outside)

    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "failed"
    assert "run_symlinks_present" in _codes(result)


def test_empty_manifest_cannot_leave_a_completed_run_passed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, _ = _write_completed_run(fixture)
    fixture.manifest_path.write_text(json.dumps({"shards": []}), encoding="utf-8")

    summary = audit_experiment(fixture.config_path)
    result = _run_result(summary, run)

    assert result["status"] == "failed"
    assert "manifest_validation_failed" in _codes(result)
    assert summary["manifests"]["training"]["status"] == "failed"


def test_missing_manifest_shard_cannot_leave_a_run_passed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, _ = _write_completed_run(fixture)
    (fixture.manifest_path.parent / "part-00098.jsonl.gz").unlink()

    summary = audit_experiment(fixture.config_path)
    result = _run_result(summary, run)
    assert result["status"] == "failed"
    assert summary["manifests"]["training"]["status"] == "failed"
    assert "manifest_validation_failed" in _codes(result)


def test_test_manifest_metadata_must_match_parent_shard_99(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    test_manifest_path = fixture.manifest_path.with_name("test_manifest.json")
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    test_manifest["shards"][0]["sha256"] = "0" * 64
    test_manifest_path.write_text(json.dumps(test_manifest), encoding="utf-8")

    summary = audit_experiment(fixture.config_path)
    assert summary["manifests"]["test"]["status"] == "failed"
    assert "manifest_parent_shard_mismatch" in {
        issue["code"] for issue in summary["manifests"]["test"]["issues"]
    }


def test_test_manifest_parent_path_must_resolve_to_dataset_manifest(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    other_dir = fixture.repository / "data" / "other"
    other_dir.mkdir()
    source = fixture.manifest_path.with_name("test_manifest.json")
    (other_dir / "test_manifest.json").write_bytes(source.read_bytes())
    text = fixture.config_path.read_text(encoding="utf-8")
    text = _replace_toml(text, "test_manifest", "data/other/test_manifest.json")
    fixture.config_path.write_text(text, encoding="utf-8")

    summary = audit_experiment(fixture.config_path)
    assert summary["manifests"]["test"]["status"] == "failed"
    assert "manifest_parent_path_unresolved" in {
        issue["code"] for issue in summary["manifests"]["test"]["issues"]
    }


def test_partial_scan_and_path_escape_are_fail_closed(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, _ = _write_completed_run(fixture)
    partial = fixture.repository / run.output_dir / "checkpoint.partial-copy"
    partial.write_bytes(b"x")
    summary = audit_experiment(fixture.config_path)
    assert summary["partial_artifacts"] == [
        f"{run.run_id}/checkpoint.partial-copy"
    ]
    assert _run_result(summary, run)["status"] == "failed"

    text = fixture.config_path.read_text(encoding="utf-8")
    text = _replace_toml(text, "output_dir", "../escaped")
    fixture.config_path.write_text(text, encoding="utf-8")
    with pytest.raises(AuditPathError):
        audit_experiment(fixture.config_path)


def test_cli_emits_machine_json_and_distinct_failure_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _make_fixture(tmp_path)
    exit_code = main(["--config", str(fixture.config_path), "--compact"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["incomplete_count"] == 30

    exit_code = main(["--config", str(tmp_path / "missing.toml"), "--compact"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "error"


def test_authenticated_checkpoint_load_always_uses_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_fixture(tmp_path)
    run, _, _ = _write_completed_run(fixture)
    original_load = torch.load
    calls: list[bool | None] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", capture)
    result = _run_result(audit_experiment(fixture.config_path), run)
    assert result["status"] == "passed"
    assert calls and calls == [True]

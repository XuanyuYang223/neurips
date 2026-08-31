"""Adversarial, read-only completion audit for Henry's formal run matrix.

The frozen TOML is treated as the launch specification.  The auditor rebuilds
the complete expected ``TrainConfig`` for every run, validates manifests and
filesystem containment, authenticates a checkpoint before loading it on CPU,
and then validates its complete training state.  It never repairs artifacts.
An unmarked run is reported as incomplete without opening its checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import tomllib
from typing import Any

import torch
from torch import Tensor

from .experiments import (
    ExperimentRun,
    _training_command,
    build_matrix,
    dataset_protocol_version,
    task_names_for_experiment,
)
from .models import build_model
from .passage import TOKEN_TO_ID
from .training import TrainConfig, build_arg_parser, parse_shard_indices


DEFAULT_CONFIG = Path("configs/henry_permutation.toml")
AUDIT_FORMAT_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_KEYS = {
    "format_version",
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "rng",
    "state",
    "config",
    "data_fingerprints",
    "validation",
}
_MARKER_KEYS = {
    "status",
    "run_id",
    "architecture",
    "tasks",
    "seed",
    "global_step",
    "epoch",
    "batches_in_epoch",
    "last_loss",
    "checkpoint",
    "checkpoint_sha256",
    "config_sha256",
    "experiment_config_sha256",
    "training_manifest_sha256",
    "validation_manifest_sha256",
    "task_accounting",
    "validation",
}
_STATE_KEYS = {
    "epoch",
    "batches_in_epoch",
    "global_step",
    "task_examples",
    "task_supervised_tokens",
    "task_loss_sum",
}
_VALIDATION_METRIC_KEYS = {
    "loss",
    "token_accuracy",
    "sequence_accuracy",
    "tokens",
    "examples",
}
_OPTIMIZER_GROUP_REQUIRED_KEYS = {
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "capturable",
    "differentiable",
    "initial_lr",
    "params",
}
_OPTIMIZER_GROUP_OPTIONAL_KEYS = {"foreach", "fused", "decoupled_weight_decay"}
_SCHEDULER_REQUIRED_KEYS = {
    "base_lrs",
    "last_epoch",
    "_step_count",
    "_get_lr_called_within_step",
    "_last_lr",
    "lr_lambdas",
}
_SCHEDULER_OPTIONAL_KEYS = {"_is_initial", "verbose"}
_MANIFEST_KEYS = {
    "base",
    "count",
    "format",
    "gzip_compresslevel",
    "gzip_mtime",
    "max_entries",
    "schema_version",
    "seed",
    "shard_count",
    "shard_size",
    "shards",
    "task_counts",
    "tasks",
    "total_bytes",
}
_SHARD_KEYS = {
    "byte_size",
    "filename",
    "first_id",
    "index",
    "last_id",
    "record_count",
    "sha256",
    "task_counts",
}


class AuditPathError(ValueError):
    """A configured artifact escaped the repository or crossed a symlink."""


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"code": code, "message": message}
    value.update(details)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_regular_nofollow(path))


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file_without_symlink(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    return stat.S_ISREG(mode)


def _read_regular_nofollow(path: Path) -> bytes:
    """Read one stable regular-file inode without following its final link."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuditPathError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        fingerprint_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        fingerprint_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if fingerprint_after != fingerprint_before or len(payload) != before.st_size:
            raise OSError(f"file changed while being read: {path}")
        current = path.lstat()
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != before.st_dev
            or current.st_ino != before.st_ino
        ):
            raise OSError(f"file path was replaced while being read: {path}")
        return payload
    finally:
        os.close(descriptor)


def _repository_for_config(config_path: Path) -> tuple[Path, Path]:
    config_absolute = _absolute(config_path)
    repository = config_absolute.parent.parent
    current = Path(config_absolute.anchor)
    for part in config_absolute.parts[1:]:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise AuditPathError(f"symlink is forbidden in config path: {current}")
    if not repository.is_dir() or repository.is_symlink():
        raise AuditPathError(f"config repository is not a real directory: {repository}")
    if not _regular_file_without_symlink(config_absolute):
        raise AuditPathError(f"config is not a regular file: {config_absolute}")
    return repository, config_absolute


def _checked_repo_path(
    raw: Any,
    repository: Path,
    *,
    label: str,
    must_exist: bool = False,
    kind: str | None = None,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AuditPathError(f"{label} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or relative.anchor or ".." in relative.parts:
        raise AuditPathError(f"{label} must stay relative to the config repository: {raw}")
    candidate = repository.joinpath(*relative.parts)
    current = repository
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise AuditPathError(f"symlink is forbidden for {label}: {current}")
    if must_exist and not os.path.lexists(candidate):
        raise AuditPathError(f"{label} does not exist: {candidate}")
    if os.path.lexists(candidate):
        mode = candidate.lstat().st_mode
        if kind == "file" and not stat.S_ISREG(mode):
            raise AuditPathError(f"{label} is not a regular file: {candidate}")
        if kind == "directory" and not stat.S_ISDIR(mode):
            raise AuditPathError(f"{label} is not a directory: {candidate}")
    return candidate


def _repo_relative(path: Path, repository: Path) -> str:
    return path.relative_to(repository).as_posix()


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _check_equal(
    issues: list[dict[str, Any]],
    *,
    code: str,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not _strict_equal(actual, expected):
        issues.append(
            _issue(code, f"{label} mismatch", expected=expected, actual=actual)
        )


def _config_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(value)
    identity.pop("resume", None)
    return identity


def training_config_sha256(value: Mapping[str, Any]) -> str:
    """Reproduce the training marker digest (which excludes resume locator)."""

    payload = json.dumps(
        _config_identity(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def expected_train_config(
    run: ExperimentRun,
    experiment: Mapping[str, Any],
    config_path: Path,
    *,
    experiment_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Rebuild the complete formal ``TrainConfig`` from the launch TOML."""

    repository, config_absolute = _repository_for_config(config_path)
    model = experiment["model"]
    training = experiment["training"]
    precision = str(training["precision"]).lower()
    config_relative = _repo_relative(config_absolute, repository)
    command = _training_command(
        run, dict(experiment), Path(config_relative)
    )
    args = build_arg_parser().parse_args(command[3:])
    model_config = json.loads(args.model_config_json)
    value = TrainConfig(
        output_dir=args.output_dir,
        manifest=args.manifest,
        train_shards=(),
        validation_manifest=args.validation_manifest,
        validation_shards=(),
        architecture=args.architecture,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        tie_embeddings=bool(model["tie_embeddings"]),
        model_config=model_config,
        tasks=tuple(value.strip() for value in args.tasks[0].split(",")),
        validation_tasks=None,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        validation_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        max_tokens_per_batch=args.max_tokens_per_batch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        max_grad_norm=args.max_grad_norm,
        shuffle_buffer_size=args.shuffle_buffer_size,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        validate_every=args.validate_every,
        validation_batches_per_task=args.validation_batches_per_task,
        device=args.device,
        amp=precision in {"bf16", "fp16", "float16", "bfloat16"},
        bf16=precision in {"bf16", "bfloat16"},
        resume=args.resume,
        shard_indices=parse_shard_indices(args.train_shard_indices),
        validation_shard_indices=parse_shard_indices(args.validation_shard_indices),
        experiment_config=args.experiment_config,
        experiment_config_sha256=(
            experiment_config_sha256
            if experiment_config_sha256 is not None
            else _sha256_file(config_absolute)
        ),
    )
    return asdict(value)


def _normalize_train_config(
    value: Mapping[str, Any],
    *,
    expected_keys: set[str],
    repository: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if set(value) != expected_keys:
        issues.append(
            _issue(
                "checkpoint_config_schema_mismatch",
                "checkpoint config keys differ from the complete TrainConfig schema",
                missing=sorted(expected_keys - set(value)),
                unexpected=sorted(set(value) - expected_keys),
            )
        )
        return None, issues
    try:
        normalized = asdict(TrainConfig.from_value(value))
    except (TypeError, ValueError) as error:
        issues.append(
            _issue(
                "checkpoint_config_invalid",
                f"checkpoint config cannot be normalized: {error}",
            )
        )
        return None, issues

    try:
        for key in ("manifest", "validation_manifest", "experiment_config"):
            normalized[key] = _repo_relative(
                _checked_repo_path(
                    normalized[key],
                    repository,
                    label=f"checkpoint config {key}",
                    must_exist=True,
                    kind="file",
                ),
                repository,
            )
        normalized["output_dir"] = _repo_relative(
            _checked_repo_path(
                normalized["output_dir"],
                repository,
                label="checkpoint config output_dir",
            ),
            repository,
        )
        for key in ("train_shards", "validation_shards"):
            normalized[key] = tuple(
                _repo_relative(
                    _checked_repo_path(
                        raw,
                        repository,
                        label=f"checkpoint config {key}",
                        must_exist=True,
                        kind="file",
                    ),
                    repository,
                )
                for raw in normalized[key]
            )
    except AuditPathError as error:
        issues.append(_issue("checkpoint_config_path_invalid", str(error)))
        return None, issues
    return normalized, issues


def _manifest_result(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": None,
        "status": "failed",
        "shard_count": None,
        "count": None,
        "issues": [],
        "_shards_by_index": {},
    }


def _audit_manifest(
    path: Path,
    *,
    expected_indices: Sequence[int],
    expected_seed: int,
    expected_schema_version: str,
    task_names: Sequence[str],
    parent_sha256: str | None = None,
    parent_path: Path | None = None,
    parent_shards: Mapping[int, Mapping[str, Any]] | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Validate manifest metadata and shard files, deferring full shard hashes."""

    result = _manifest_result(path)
    issues: list[dict[str, Any]] = result["issues"]
    if not _regular_file_without_symlink(path):
        issues.append(_issue("manifest_not_regular", "manifest is missing or symlinked"))
        return result
    try:
        payload = _read_regular_nofollow(path)
        manifest = json.loads(payload)
    except (OSError, AuditPathError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(_issue("manifest_unreadable", f"manifest cannot be read: {error}"))
        return result
    result["sha256"] = _sha256_bytes(payload)
    if not isinstance(manifest, Mapping):
        issues.append(_issue("manifest_not_object", "manifest is not a JSON object"))
        return result

    expected_keyset = _MANIFEST_KEYS | (
        {"parent_manifest", "parent_manifest_sha256", "split"} if split else set()
    )
    if set(manifest) != expected_keyset:
        issues.append(
            _issue(
                "manifest_schema_mismatch",
                "manifest keys differ from the formal schema",
                missing=sorted(expected_keyset - set(manifest)),
                unexpected=sorted(set(manifest) - expected_keyset),
            )
        )

    indices = tuple(expected_indices)
    shard_size = 100_000
    per_task_per_shard = shard_size // len(task_names)
    expected_task_counts = {
        task: per_task_per_shard * len(indices) for task in task_names
    }
    for key, expected in (
        ("base", 100),
        ("count", shard_size * len(indices)),
        ("format", "jsonl.gz"),
        ("gzip_compresslevel", 6),
        ("gzip_mtime", 0),
        ("max_entries", 30),
        ("schema_version", expected_schema_version),
        ("seed", expected_seed),
        ("shard_count", len(indices)),
        ("shard_size", shard_size),
        ("tasks", list(task_names)),
        ("task_counts", expected_task_counts),
    ):
        _check_equal(
            issues,
            code=f"manifest_{key}_mismatch",
            label=f"manifest {key}",
            actual=manifest.get(key),
            expected=expected,
        )
    result["shard_count"] = manifest.get("shard_count")
    result["count"] = manifest.get("count")

    if split:
        _check_equal(
            issues,
            code="manifest_split_mismatch",
            label="test manifest split",
            actual=manifest.get("split"),
            expected=split,
        )
        _check_equal(
            issues,
            code="manifest_parent_sha256_mismatch",
            label="test manifest parent SHA-256",
            actual=manifest.get("parent_manifest_sha256"),
            expected=parent_sha256,
        )
        _check_equal(
            issues,
            code="manifest_parent_path_mismatch",
            label="test manifest parent path",
            actual=manifest.get("parent_manifest"),
            expected="manifest.json",
        )
        if parent_path is None or path.parent / "manifest.json" != parent_path:
            issues.append(
                _issue(
                    "manifest_parent_path_unresolved",
                    "test manifest parent reference does not resolve to dataset_manifest",
                )
            )

    shards = manifest.get("shards")
    if not isinstance(shards, list):
        issues.append(_issue("manifest_shards_invalid", "manifest shards is not a list"))
        return result
    if len(shards) != len(indices):
        issues.append(
            _issue(
                "manifest_shard_count_mismatch",
                "manifest shard list has the wrong length",
                expected=len(indices),
                actual=len(shards),
            )
        )

    total_bytes = 0
    for position, expected_index in enumerate(indices):
        if position >= len(shards):
            break
        shard = shards[position]
        if not isinstance(shard, Mapping):
            issues.append(
                _issue(
                    "manifest_shard_invalid",
                    "shard metadata is not an object",
                    position=position,
                )
            )
            continue
        result["_shards_by_index"][expected_index] = dict(shard)
        if set(shard) != _SHARD_KEYS:
            issues.append(
                _issue(
                    "manifest_shard_schema_mismatch",
                    "shard metadata keys differ from the formal schema",
                    position=position,
                    missing=sorted(_SHARD_KEYS - set(shard)),
                    unexpected=sorted(set(shard) - _SHARD_KEYS),
                )
            )
        filename = f"part-{expected_index:05d}.jsonl.gz"
        expected_values = {
            "index": expected_index,
            "filename": filename,
            "first_id": expected_index * shard_size,
            "last_id": (expected_index + 1) * shard_size - 1,
            "record_count": shard_size,
            "task_counts": {
                task: per_task_per_shard for task in task_names
            },
        }
        for key, expected in expected_values.items():
            _check_equal(
                issues,
                code=f"manifest_shard_{key}_mismatch",
                label=f"shard {expected_index} {key}",
                actual=shard.get(key),
                expected=expected,
            )
        if parent_shards is not None:
            parent_shard = parent_shards.get(expected_index)
            if parent_shard is None or not _strict_equal(shard, parent_shard):
                issues.append(
                    _issue(
                        "manifest_parent_shard_mismatch",
                        "split shard metadata differs from the parent manifest",
                        index=expected_index,
                    )
                )
        digest = shard.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            issues.append(
                _issue(
                    "manifest_shard_sha256_invalid",
                    "shard SHA-256 metadata is invalid",
                    index=expected_index,
                )
            )
        byte_size = shard.get("byte_size")
        if type(byte_size) is not int or byte_size <= 0:
            issues.append(
                _issue(
                    "manifest_shard_byte_size_invalid",
                    "shard byte_size must be a positive integer",
                    index=expected_index,
                    actual=byte_size,
                )
            )
            continue
        total_bytes += byte_size
        shard_path = path.parent / filename
        if not _regular_file_without_symlink(shard_path):
            issues.append(
                _issue(
                    "manifest_shard_not_regular",
                    "shard is missing or symlinked",
                    index=expected_index,
                    path=str(shard_path),
                )
            )
            continue
        try:
            shard_stat = shard_path.lstat()
        except OSError as error:
            issues.append(
                _issue(
                    "manifest_shard_stat_failed",
                    f"could not stat shard: {error}",
                    index=expected_index,
                )
            )
        else:
            actual_size = shard_stat.st_size
            if actual_size != byte_size:
                issues.append(
                    _issue(
                        "manifest_shard_size_mismatch",
                        "shard size differs from manifest metadata",
                        index=expected_index,
                        expected=byte_size,
                        actual=actual_size,
                    )
                )

    _check_equal(
        issues,
        code="manifest_total_bytes_mismatch",
        label="manifest total_bytes",
        actual=manifest.get("total_bytes"),
        expected=total_bytes,
    )
    result["status"] = "passed" if not issues else "failed"
    return result


def _is_partial_name(name: str) -> bool:
    lowered = name.lower()
    return ".tmp" in lowered or "partial" in lowered or lowered.endswith(".part")


def _scan_output_tree(root: Path) -> dict[str, Any]:
    """Scan without following links; any link or scan error makes the audit fail."""

    result: dict[str, Any] = {"partials": [], "symlinks": [], "issues": []}
    if not os.path.lexists(root):
        return result
    if root.is_symlink() or not root.is_dir():
        result["issues"].append(
            _issue("output_root_invalid", "output root is symlinked or not a directory")
        )
        if root.is_symlink():
            result["symlinks"].append(root)
        return result
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if _is_partial_name(entry.name):
                        result["partials"].append(path)
                    try:
                        if entry.is_symlink():
                            result["symlinks"].append(path)
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(path)
                    except OSError as error:
                        result["issues"].append(
                            _issue(
                                "output_entry_scan_failed",
                                f"could not inspect output entry: {error}",
                                path=str(path),
                            )
                        )
        except OSError as error:
            result["issues"].append(
                _issue(
                    "output_directory_scan_failed",
                    f"could not scan output directory: {error}",
                    path=str(directory),
                )
            )
    result["partials"].sort(key=str)
    result["symlinks"].sort(key=str)
    if result["partials"]:
        result["issues"].append(
            _issue("partial_artifacts_present", "temporary/partial artifacts remain")
        )
    if result["symlinks"]:
        result["issues"].append(
            _issue("output_symlinks_present", "symlinks are forbidden below output root")
        )
    return result


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _iter_values(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_values(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_values(child, f"{path}[{index}]")
    else:
        yield path, value


def _finite_summary(value: Any, *, root_name: str) -> dict[str, Any]:
    tensor_count = 0
    tensor_elements = 0
    scalar_number_count = 0
    nonfinite: list[dict[str, Any]] = []
    check_errors: list[dict[str, Any]] = []
    for path, leaf in _iter_values(value, root_name):
        if isinstance(leaf, Tensor):
            tensor_count += 1
            tensor_elements += leaf.numel()
            try:
                mask = torch.isfinite(leaf)
                if mask.layout != torch.strided:
                    mask = mask.to_dense()
                bad = int((~mask).sum().item())
            except (RuntimeError, TypeError, NotImplementedError) as error:
                check_errors.append({"path": path, "error": str(error)})
                continue
            if bad:
                nonfinite.append(
                    {
                        "path": path,
                        "kind": "tensor",
                        "dtype": str(leaf.dtype),
                        "shape": list(leaf.shape),
                        "nonfinite_elements": bad,
                    }
                )
        elif isinstance(leaf, float):
            scalar_number_count += 1
            if not math.isfinite(leaf):
                nonfinite.append({"path": path, "kind": "scalar", "value": repr(leaf)})
        elif isinstance(leaf, complex):
            scalar_number_count += 1
            if not (math.isfinite(leaf.real) and math.isfinite(leaf.imag)):
                nonfinite.append({"path": path, "kind": "scalar", "value": repr(leaf)})
        elif isinstance(leaf, int) and not isinstance(leaf, bool):
            scalar_number_count += 1
    return {
        "status": "passed" if not nonfinite and not check_errors else "failed",
        "tensor_count": tensor_count,
        "tensor_elements": tensor_elements,
        "scalar_number_count": scalar_number_count,
        "nonfinite": nonfinite,
        "check_errors": check_errors,
    }


def _combine_finite_summaries(*values: Mapping[str, Any]) -> dict[str, Any]:
    nonfinite = [item for value in values for item in value["nonfinite"]]
    check_errors = [item for value in values for item in value["check_errors"]]
    return {
        "status": "passed" if not nonfinite and not check_errors else "failed",
        "tensor_count": sum(value["tensor_count"] for value in values),
        "tensor_elements": sum(value["tensor_elements"] for value in values),
        "scalar_number_count": sum(value["scalar_number_count"] for value in values),
        "nonfinite": nonfinite,
        "check_errors": check_errors,
    }


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(config["model_config"])
    kwargs.update(
        {
            "model_type": config["architecture"],
            "vocab_size": len(TOKEN_TO_ID),
            "max_seq_len": config["max_seq_len"],
            "d_model": config["d_model"],
            "layers": config["num_layers"],
            "dropout": config["dropout"],
            "mlp_ratio": config["mlp_ratio"],
            "tie_embeddings": config["tie_embeddings"],
        }
    )
    if config["architecture"] == "transformer":
        kwargs["n_heads"] = config["num_heads"]
    return kwargs


def _validate_model(
    model_state: Any,
    *,
    expected_config: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    if not isinstance(model_state, Mapping) or not model_state:
        issues.append(_issue("checkpoint_model_empty", "model state must be non-empty"))
        return []
    model = build_model(**_model_kwargs(expected_config))
    expected_state = model.state_dict()
    if set(model_state) != set(expected_state):
        issues.append(
            _issue(
                "model_state_schema_mismatch",
                "model state keys differ from the expected architecture",
                missing=sorted(set(expected_state) - set(model_state)),
                unexpected=sorted(set(model_state) - set(expected_state)),
            )
        )
    for key in set(model_state) & set(expected_state):
        actual = model_state[key]
        expected = expected_state[key]
        if not isinstance(actual, Tensor):
            issues.append(
                _issue("model_value_not_tensor", "model state value is not a tensor", key=key)
            )
        elif actual.shape != expected.shape or actual.dtype != expected.dtype:
            issues.append(
                _issue(
                    "model_tensor_schema_mismatch",
                    "model tensor shape or dtype differs from expected",
                    key=key,
                    expected_shape=list(expected.shape),
                    actual_shape=list(actual.shape),
                    expected_dtype=str(expected.dtype),
                    actual_dtype=str(actual.dtype),
                )
            )
    if expected_config["tie_embeddings"] and {
        "token_embedding.weight",
        "lm_head.weight",
    } <= set(model_state):
        token_weight = model_state["token_embedding.weight"]
        head_weight = model_state["lm_head.weight"]
        if not (
            isinstance(token_weight, Tensor)
            and isinstance(head_weight, Tensor)
            and torch.equal(token_weight, head_weight)
        ):
            issues.append(
                _issue(
                    "tied_embedding_mismatch",
                    "tied token_embedding and lm_head checkpoint tensors differ",
                )
            )
    try:
        model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError) as error:
        issues.append(
            _issue(
                "model_strict_load_failed",
                f"expected build_model rejected the state dict: {error}",
            )
        )
    parameter_specs = [
        (tuple(parameter.shape), parameter.dtype) for parameter in model.parameters()
    ]
    del model
    return parameter_specs


def _validate_optimizer(
    optimizer: Any,
    *,
    parameter_specs: Sequence[tuple[tuple[int, ...], torch.dtype]],
    expected_config: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(optimizer, Mapping) or set(optimizer) != {"state", "param_groups"}:
        issues.append(_issue("optimizer_schema_mismatch", "optimizer schema is invalid"))
        return
    states = optimizer.get("state")
    groups = optimizer.get("param_groups")
    expected_ids = set(range(len(parameter_specs)))
    if not isinstance(states, Mapping) or set(states) != expected_ids:
        issues.append(
            _issue(
                "optimizer_state_ids_mismatch",
                "optimizer states do not cover every unique model parameter",
                expected_count=len(expected_ids),
                actual_count=len(states) if isinstance(states, Mapping) else None,
            )
        )
    else:
        for parameter_id, (shape, dtype) in enumerate(parameter_specs):
            state = states[parameter_id]
            if not isinstance(state, Mapping) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                issues.append(
                    _issue(
                        "optimizer_parameter_state_schema_mismatch",
                        "AdamW parameter state schema is invalid",
                        parameter_id=parameter_id,
                    )
                )
                continue
            step = state["step"]
            valid_step = (
                isinstance(step, Tensor)
                and step.shape == torch.Size([])
                and step.dtype == torch.float32
            )
            if valid_step:
                try:
                    valid_step = float(step) == float(expected_config["max_steps"])
                except (RuntimeError, TypeError, ValueError):
                    valid_step = False
            if not valid_step:
                issues.append(
                    _issue(
                        "optimizer_step_mismatch",
                        "optimizer parameter step differs from max_steps",
                        parameter_id=parameter_id,
                    )
                )
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = state[name]
                if (
                    not isinstance(tensor, Tensor)
                    or tuple(tensor.shape) != shape
                    or tensor.dtype != dtype
                ):
                    issues.append(
                        _issue(
                            "optimizer_tensor_schema_mismatch",
                            "optimizer moment shape/dtype differs from its parameter",
                            parameter_id=parameter_id,
                            tensor=name,
                        )
                    )
                elif name == "exp_avg_sq" and bool((tensor < 0).any()):
                    issues.append(
                        _issue(
                            "optimizer_second_moment_negative",
                            "AdamW exp_avg_sq cannot contain negative values",
                            parameter_id=parameter_id,
                        )
                    )
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        issues.append(_issue("optimizer_param_groups_invalid", "expected one AdamW group"))
        return
    group = groups[0]
    group_keys = set(group)
    if not _OPTIMIZER_GROUP_REQUIRED_KEYS <= group_keys or not group_keys <= (
        _OPTIMIZER_GROUP_REQUIRED_KEYS | _OPTIMIZER_GROUP_OPTIONAL_KEYS
    ):
        issues.append(
            _issue(
                "optimizer_param_group_schema_mismatch",
                "AdamW parameter group keys differ from formal schema",
                missing=sorted(_OPTIMIZER_GROUP_REQUIRED_KEYS - group_keys),
                unexpected=sorted(
                    group_keys
                    - _OPTIMIZER_GROUP_REQUIRED_KEYS
                    - _OPTIMIZER_GROUP_OPTIONAL_KEYS
                ),
            )
        )
    expected_values = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": expected_config["weight_decay"],
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "initial_lr": expected_config["learning_rate"],
        "params": list(range(len(parameter_specs))),
    }
    for key, expected in expected_values.items():
        _check_equal(
            issues,
            code=f"optimizer_{key}_mismatch",
            label=f"optimizer {key}",
            actual=group.get(key),
            expected=expected,
        )
    for key, expected in (
        ("foreach", None),
        ("fused", None),
        ("decoupled_weight_decay", True),
    ):
        if key in group:
            _check_equal(
                issues,
                code=f"optimizer_{key}_mismatch",
                label=f"optimizer {key}",
                actual=group[key],
                expected=expected,
            )
    final_lr = expected_config["learning_rate"] * expected_config["min_lr_ratio"]
    lr = group.get("lr")
    if not isinstance(lr, float) or not math.isclose(lr, final_lr, rel_tol=1e-12):
        issues.append(
            _issue("optimizer_lr_mismatch", "optimizer final learning rate is incorrect")
        )


def _validate_scheduler(
    scheduler: Any,
    *,
    optimizer: Any,
    expected_config: Mapping[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(scheduler, Mapping):
        issues.append(_issue("scheduler_schema_mismatch", "scheduler schema is invalid"))
        return
    scheduler_keys = set(scheduler)
    if not _SCHEDULER_REQUIRED_KEYS <= scheduler_keys or not scheduler_keys <= (
        _SCHEDULER_REQUIRED_KEYS | _SCHEDULER_OPTIONAL_KEYS
    ):
        issues.append(
            _issue(
                "scheduler_schema_mismatch",
                "scheduler schema is invalid",
                missing=sorted(_SCHEDULER_REQUIRED_KEYS - scheduler_keys),
                unexpected=sorted(
                    scheduler_keys
                    - _SCHEDULER_REQUIRED_KEYS
                    - _SCHEDULER_OPTIONAL_KEYS
                ),
            )
        )
        return
    expected_steps = expected_config["max_steps"]
    for key, expected in (
        ("base_lrs", [expected_config["learning_rate"]]),
        ("last_epoch", expected_steps),
        ("_step_count", expected_steps + 1),
        ("_get_lr_called_within_step", False),
        ("lr_lambdas", [None]),
    ):
        _check_equal(
            issues,
            code=f"scheduler_{key}_mismatch",
            label=f"scheduler {key}",
            actual=scheduler.get(key),
            expected=expected,
        )
    if "_is_initial" in scheduler:
        _check_equal(
            issues,
            code="scheduler__is_initial_mismatch",
            label="scheduler _is_initial",
            actual=scheduler["_is_initial"],
            expected=False,
        )
    if "verbose" in scheduler:
        _check_equal(
            issues,
            code="scheduler_verbose_mismatch",
            label="scheduler verbose",
            actual=scheduler["verbose"],
            expected=False,
        )
    try:
        optimizer_lr = optimizer["param_groups"][0]["lr"]
    except (KeyError, IndexError, TypeError):
        optimizer_lr = None
    if not _strict_equal(scheduler.get("_last_lr"), [optimizer_lr]):
        issues.append(
            _issue("scheduler_last_lr_mismatch", "scheduler and optimizer LR disagree")
        )


def _validate_rng(rng: Any, issues: list[dict[str, Any]]) -> None:
    if not isinstance(rng, Mapping) or set(rng) != {"python", "torch", "cuda"}:
        issues.append(_issue("rng_schema_mismatch", "RNG state schema is invalid"))
        return
    python_rng = rng["python"]
    try:
        random.Random().setstate(python_rng)
    except (TypeError, ValueError):
        issues.append(_issue("python_rng_invalid", "Python RNG state is invalid"))
    torch_rng = rng["torch"]
    if (
        not isinstance(torch_rng, Tensor)
        or torch_rng.dtype != torch.uint8
        or torch_rng.ndim != 1
        or not torch_rng.numel()
    ):
        issues.append(_issue("torch_rng_invalid", "Torch RNG state is invalid"))
    else:
        try:
            torch.Generator(device="cpu").set_state(torch_rng)
        except RuntimeError:
            issues.append(_issue("torch_rng_invalid", "Torch RNG state cannot be restored"))
    cuda_rng = rng["cuda"]
    if not isinstance(cuda_rng, list) or len(cuda_rng) != 1 or not all(
        isinstance(value, Tensor)
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() == 16
        for value in cuda_rng
    ):
        issues.append(_issue("cuda_rng_invalid", "CUDA RNG state is invalid"))


def _validate_validation(
    value: Any,
    issues: list[dict[str, Any]],
    *,
    label: str,
    task_names: Sequence[str],
) -> None:
    expected_tasks = set(task_names)
    if not isinstance(value, Mapping) or set(value) != expected_tasks:
        issues.append(
            _issue(
                f"{label}_tasks_mismatch",
                f"{label} must contain all 20 validation tasks",
            )
        )
        return
    for task, metrics in value.items():
        if not isinstance(metrics, Mapping) or set(metrics) != _VALIDATION_METRIC_KEYS:
            issues.append(
                _issue(
                    f"{label}_metric_schema_mismatch",
                    f"{label} metric schema is invalid",
                    task=task,
                )
            )
            continue
        loss = metrics["loss"]
        if not isinstance(loss, float) or not math.isfinite(loss) or loss < 0:
            issues.append(_issue(f"{label}_loss_invalid", "validation loss is invalid", task=task))
        for key in ("token_accuracy", "sequence_accuracy"):
            accuracy = metrics[key]
            if (
                not isinstance(accuracy, float)
                or not math.isfinite(accuracy)
                or not 0.0 <= accuracy <= 1.0
            ):
                issues.append(
                    _issue(f"{label}_{key}_invalid", "validation accuracy is invalid", task=task)
                )
        for key in ("tokens", "examples"):
            count = metrics[key]
            if type(count) is not int or count <= 0:
                issues.append(
                    _issue(f"{label}_{key}_invalid", "validation count is invalid", task=task)
                )


def _validate_state_and_accounting(
    state: Any,
    marker: Mapping[str, Any],
    *,
    run: ExperimentRun,
    expected_steps: int,
    issues: list[dict[str, Any]],
) -> None:
    if not isinstance(state, Mapping) or set(state) != _STATE_KEYS:
        issues.append(_issue("checkpoint_state_schema_mismatch", "training state schema is invalid"))
        return
    for key in ("epoch", "batches_in_epoch", "global_step"):
        value = state[key]
        if type(value) is not int or value < 0:
            issues.append(_issue(f"checkpoint_state_{key}_invalid", f"state {key} is invalid"))
    _check_equal(
        issues,
        code="checkpoint_global_step_mismatch",
        label="checkpoint global_step",
        actual=state.get("global_step"),
        expected=expected_steps,
    )
    for key in ("epoch", "batches_in_epoch"):
        _check_equal(
            issues,
            code=f"marker_{key}_state_mismatch",
            label=f"marker/checkpoint {key}",
            actual=marker.get(key),
            expected=state.get(key),
        )
    task_set = set(run.tasks)
    examples = state.get("task_examples")
    tokens = state.get("task_supervised_tokens")
    loss_sums = state.get("task_loss_sum")
    for name, mapping in (
        ("task_examples", examples),
        ("task_supervised_tokens", tokens),
        ("task_loss_sum", loss_sums),
    ):
        if not isinstance(mapping, Mapping) or set(mapping) != task_set:
            issues.append(
                _issue("checkpoint_task_state_mismatch", f"state {name} task keys are invalid")
            )
    accounting = marker.get("task_accounting")
    if not isinstance(accounting, Mapping) or set(accounting) != task_set:
        issues.append(
            _issue("marker_task_accounting_missing", "marker task_accounting is missing or incomplete")
        )
        return
    if not all(isinstance(value, Mapping) for value in (examples, tokens, loss_sums)):
        return
    for task in run.tasks:
        record = accounting[task]
        if not isinstance(record, Mapping) or set(record) != {
            "examples",
            "supervised_tokens",
            "mean_example_loss",
        }:
            issues.append(
                _issue("marker_task_accounting_schema_mismatch", "task accounting schema is invalid", task=task)
            )
            continue
        example_count = examples[task]
        token_count = tokens[task]
        loss_sum = loss_sums[task]
        if type(example_count) is not int or example_count <= 0:
            issues.append(_issue("state_task_examples_invalid", "task example count is invalid", task=task))
            continue
        if type(token_count) is not int or token_count <= 0:
            issues.append(_issue("state_task_tokens_invalid", "task token count is invalid", task=task))
        elif token_count < example_count:
            issues.append(
                _issue(
                    "state_task_tokens_too_small",
                    "each example must contribute at least one supervised token",
                    task=task,
                )
            )
        if not isinstance(loss_sum, float) or not math.isfinite(loss_sum) or loss_sum < 0:
            issues.append(
                _issue(
                    "state_task_loss_invalid",
                    "task loss sum must be finite and nonnegative",
                    task=task,
                )
            )
        _check_equal(
            issues,
            code="marker_task_examples_mismatch",
            label=f"{task} example count",
            actual=record.get("examples"),
            expected=example_count,
        )
        _check_equal(
            issues,
            code="marker_task_tokens_mismatch",
            label=f"{task} supervised token count",
            actual=record.get("supervised_tokens"),
            expected=token_count,
        )
        mean = record.get("mean_example_loss")
        expected_mean = (
            loss_sum / example_count
            if isinstance(loss_sum, float) and math.isfinite(loss_sum) and loss_sum >= 0
            else None
        )
        if (
            not isinstance(mean, float)
            or not math.isfinite(mean)
            or mean < 0
            or expected_mean is None
            or not math.isclose(mean, expected_mean, rel_tol=1e-12, abs_tol=0.0)
        ):
            issues.append(
                _issue("marker_task_loss_mismatch", "task mean loss disagrees with state", task=task)
            )


def _run_artifacts(
    artifacts: Sequence[Path], output_dir: Path, output_root: Path
) -> list[str]:
    return [
        _display_path(path, output_root)
        for path in artifacts
        if path == output_dir or output_dir in path.parents
    ]


def audit_run(
    run: ExperimentRun,
    *,
    output_dir: Path,
    output_root: Path,
    repository: Path,
    expected_config: Mapping[str, Any],
    experiment_config_sha256: str,
    training_manifest_sha256: str | None,
    validation_manifest_sha256: str | None,
    manifests_ok: bool,
    validation_tasks: Sequence[str],
    partial_artifacts: Sequence[Path] = (),
    symlinks: Sequence[Path] = (),
) -> dict[str, Any]:
    """Audit one formal run without mutating or following links."""

    marker_path = output_dir / "completed.json"
    checkpoint_path = output_dir / "checkpoint.pt"
    expected_steps = int(expected_config["max_steps"])
    issues: list[dict[str, Any]] = []
    run_partials = _run_artifacts(partial_artifacts, output_dir, output_root)
    run_symlinks = _run_artifacts(symlinks, output_dir, output_root)
    result: dict[str, Any] = {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "tasks": list(run.tasks),
        "seed": run.seed,
        "output_dir": _display_path(output_dir, repository),
        "marker_path": _display_path(marker_path, repository),
        "checkpoint_path": _display_path(checkpoint_path, repository),
        "marker_present": _regular_file_without_symlink(marker_path),
        "checkpoint_present": _regular_file_without_symlink(checkpoint_path),
        "expected_global_step": expected_steps,
        "marker_global_step": None,
        "checkpoint_global_step": None,
        "marker_checkpoint_sha256": None,
        "checkpoint_sha256": None,
        "checkpoint_size_bytes": None,
        "checkpoint_numeric": {"status": "not_checked"},
        "model_tensors": {"status": "not_checked"},
        "optimizer_tensors": {"status": "not_checked"},
        "results": None,
        "partial_artifacts": run_partials,
        "symlinks": run_symlinks,
        "issues": issues,
    }
    if not manifests_ok:
        issues.append(
            _issue("manifest_validation_failed", "required manifest validation did not pass")
        )
    if run_partials:
        issues.append(_issue("partial_artifacts_present", "run has partial artifacts"))
    if run_symlinks:
        issues.append(_issue("run_symlinks_present", "run contains a forbidden symlink"))
        result["status"] = "failed"
        return result

    if not os.path.lexists(marker_path):
        issues.append(_issue("completion_marker_missing", "completed.json is missing"))
        result["status"] = "incomplete"
        return result
    if not _regular_file_without_symlink(marker_path):
        issues.append(_issue("completion_marker_not_regular", "completed.json is symlinked or irregular"))
        result["status"] = "failed"
        return result
    try:
        marker = json.loads(_read_regular_nofollow(marker_path))
    except (OSError, AuditPathError, UnicodeError, json.JSONDecodeError) as error:
        issues.append(_issue("completion_marker_unreadable", f"cannot parse marker: {error}"))
        result["status"] = "failed"
        return result
    if not isinstance(marker, Mapping):
        issues.append(_issue("completion_marker_not_object", "completed.json is not an object"))
        result["status"] = "failed"
        return result
    if set(marker) != _MARKER_KEYS:
        issues.append(
            _issue(
                "completion_marker_schema_mismatch",
                "completion marker keys differ from formal schema",
                missing=sorted(_MARKER_KEYS - set(marker)),
                unexpected=sorted(set(marker) - _MARKER_KEYS),
            )
        )
    if "task_accounting" not in marker:
        issues.append(_issue("marker_task_accounting_missing", "marker lacks task_accounting"))
    if "validation" not in marker:
        issues.append(_issue("marker_validation_missing", "marker lacks validation"))
    result["results"] = {
        key: marker.get(key)
        for key in (
            "epoch",
            "batches_in_epoch",
            "last_loss",
            "task_accounting",
            "validation",
        )
    }
    result["marker_global_step"] = marker.get("global_step")
    result["marker_checkpoint_sha256"] = marker.get("checkpoint_sha256")
    for key, expected in (
        ("status", "completed"),
        ("run_id", run.run_id),
        ("architecture", run.architecture),
        ("tasks", list(run.tasks)),
        ("seed", run.seed),
        ("global_step", expected_steps),
        ("experiment_config_sha256", experiment_config_sha256),
        ("training_manifest_sha256", training_manifest_sha256),
        ("validation_manifest_sha256", validation_manifest_sha256),
    ):
        _check_equal(
            issues,
            code=f"marker_{key}_mismatch",
            label=f"marker {key}",
            actual=marker.get(key),
            expected=expected,
        )
    last_loss = marker.get("last_loss")
    if (
        not isinstance(last_loss, float)
        or not math.isfinite(last_loss)
        or last_loss < 0
    ):
        issues.append(_issue("marker_last_loss_invalid", "marker last_loss is not finite/nonnegative"))

    try:
        referenced_checkpoint = _checked_repo_path(
            marker.get("checkpoint"),
            repository,
            label="marker checkpoint",
            must_exist=True,
            kind="file",
        )
    except AuditPathError as error:
        issues.append(_issue("marker_checkpoint_path_invalid", str(error)))
        referenced_checkpoint = None
    if referenced_checkpoint != checkpoint_path:
        issues.append(
            _issue(
                "marker_checkpoint_path_mismatch",
                "marker does not reference the canonical run checkpoint",
            )
        )
    if marker.get("status") != "completed":
        result["status"] = "failed"
        return result
    if not _regular_file_without_symlink(checkpoint_path):
        issues.append(_issue("checkpoint_not_regular", "checkpoint is missing, symlinked, or irregular"))
        result["status"] = "failed"
        return result
    marker_digest = marker.get("checkpoint_sha256")
    if not isinstance(marker_digest, str) or not _SHA256_RE.fullmatch(marker_digest):
        issues.append(_issue("marker_checkpoint_sha256_invalid", "checkpoint SHA-256 is malformed"))
        result["status"] = "failed"
        return result
    try:
        checkpoint_bytes = _read_regular_nofollow(checkpoint_path)
        result["checkpoint_size_bytes"] = len(checkpoint_bytes)
        digest_before = _sha256_bytes(checkpoint_bytes)
    except (OSError, AuditPathError) as error:
        issues.append(_issue("checkpoint_unreadable", f"cannot hash checkpoint: {error}"))
        result["status"] = "failed"
        return result
    result["checkpoint_sha256"] = digest_before
    if digest_before != marker_digest:
        issues.append(_issue("checkpoint_sha256_mismatch", "checkpoint does not match marker SHA-256"))
        result["status"] = "failed"
        return result
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        issues.append(
            _issue(
                "checkpoint_deserialization_failed",
                f"weights_only checkpoint load failed: {type(error).__name__}: {error}",
            )
        )
        result["status"] = "failed"
        return result
    if not isinstance(checkpoint, Mapping):
        issues.append(_issue("checkpoint_not_mapping", "checkpoint is not a mapping"))
        result["status"] = "failed"
        return result
    if set(checkpoint) != _CHECKPOINT_KEYS:
        issues.append(
            _issue(
                "checkpoint_schema_mismatch",
                "checkpoint keys differ from formal schema",
                missing=sorted(_CHECKPOINT_KEYS - set(checkpoint)),
                unexpected=sorted(set(checkpoint) - _CHECKPOINT_KEYS),
            )
        )
    _check_equal(
        issues,
        code="checkpoint_format_version_mismatch",
        label="checkpoint format_version",
        actual=checkpoint.get("format_version"),
        expected=1,
    )
    model_numeric = _finite_summary(checkpoint.get("model"), root_name="checkpoint.model")
    optimizer_numeric = _finite_summary(
        checkpoint.get("optimizer"), root_name="checkpoint.optimizer"
    )
    remaining_numeric = _finite_summary(
        {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "optimizer"}
        },
        root_name="checkpoint",
    )
    numeric = _combine_finite_summaries(
        model_numeric, optimizer_numeric, remaining_numeric
    )
    result["checkpoint_numeric"] = numeric
    if numeric["status"] != "passed":
        issues.append(_issue("checkpoint_numbers_not_finite", "checkpoint contains non-finite numbers"))
    result["model_tensors"] = model_numeric
    result["optimizer_tensors"] = optimizer_numeric

    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, Mapping):
        normalized_actual, config_issues = _normalize_train_config(
            checkpoint_config,
            expected_keys=set(expected_config),
            repository=repository,
        )
        issues.extend(config_issues)
        normalized_expected, expected_issues = _normalize_train_config(
            expected_config,
            expected_keys=set(expected_config),
            repository=repository,
        )
        issues.extend(expected_issues)
        if normalized_actual is not None and normalized_expected is not None:
            differing = sorted(
                key
                for key in expected_config
                if not _strict_equal(normalized_actual[key], normalized_expected[key])
            )
            if differing:
                issues.append(
                    _issue(
                        "checkpoint_full_config_mismatch",
                        "checkpoint TrainConfig differs from reconstructed launch config",
                        fields=differing,
                    )
                )
        expected_config_digest = training_config_sha256(expected_config)
        try:
            actual_config_digest = training_config_sha256(checkpoint_config)
        except (TypeError, ValueError) as error:
            issues.append(
                _issue(
                    "checkpoint_config_unhashable",
                    f"checkpoint config is not canonical JSON: {error}",
                )
            )
        else:
            _check_equal(
                issues,
                code="checkpoint_config_sha256_mismatch",
                label="checkpoint config SHA-256 versus launch spec",
                actual=actual_config_digest,
                expected=expected_config_digest,
            )
        _check_equal(
            issues,
            code="marker_config_sha256_mismatch",
            label="marker config SHA-256",
            actual=marker.get("config_sha256"),
            expected=expected_config_digest,
        )
    else:
        issues.append(_issue("checkpoint_config_missing", "checkpoint config is missing"))

    parameter_specs = _validate_model(
        checkpoint.get("model"), expected_config=expected_config, issues=issues
    )
    _validate_optimizer(
        checkpoint.get("optimizer"),
        parameter_specs=parameter_specs,
        expected_config=expected_config,
        issues=issues,
    )
    _validate_scheduler(
        checkpoint.get("scheduler"),
        optimizer=checkpoint.get("optimizer"),
        expected_config=expected_config,
        issues=issues,
    )
    scaler = checkpoint.get("scaler")
    if not isinstance(scaler, Mapping) or (expected_config["bf16"] and scaler):
        issues.append(_issue("scaler_schema_mismatch", "GradScaler state is invalid for precision"))
    _validate_rng(checkpoint.get("rng"), issues)

    fingerprints = checkpoint.get("data_fingerprints")
    expected_fingerprints = {
        "training_manifest_sha256": training_manifest_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
    }
    _check_equal(
        issues,
        code="checkpoint_data_fingerprints_mismatch",
        label="checkpoint data fingerprints",
        actual=fingerprints,
        expected=expected_fingerprints,
    )
    state = checkpoint.get("state")
    if isinstance(state, Mapping):
        result["checkpoint_global_step"] = state.get("global_step")
    _validate_state_and_accounting(
        state,
        marker,
        run=run,
        expected_steps=expected_steps,
        issues=issues,
    )
    checkpoint_validation = checkpoint.get("validation")
    _validate_validation(
        checkpoint_validation,
        issues,
        label="checkpoint_validation",
        task_names=validation_tasks,
    )
    marker_validation = marker.get("validation")
    _validate_validation(
        marker_validation,
        issues,
        label="marker_validation",
        task_names=validation_tasks,
    )
    if not _strict_equal(marker_validation, checkpoint_validation):
        issues.append(
            _issue("marker_validation_checkpoint_mismatch", "marker and checkpoint validation differ")
        )
    result["status"] = "passed" if not issues else "failed"
    return result


def _strict_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _strict_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(child) for child in value]
    return value


def audit_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return a strict-JSON-compatible audit of all 30 expected formal runs."""

    repository, config_absolute = _repository_for_config(Path(config_path))
    config_bytes = _read_regular_nofollow(config_absolute)
    experiment = tomllib.loads(config_bytes.decode("utf-8"))
    config_sha256 = _sha256_bytes(config_bytes)
    expected_schema_version = dataset_protocol_version(experiment)
    task_names = task_names_for_experiment(experiment)
    runs = build_matrix(config_absolute)
    global_issues: list[dict[str, Any]] = []
    if len(runs) != 30:
        global_issues.append(
            _issue("expected_run_count_mismatch", "formal matrix must contain exactly 30 runs")
        )

    try:
        dataset_manifest = _checked_repo_path(
            experiment["dataset_manifest"],
            repository,
            label="dataset_manifest",
            must_exist=True,
            kind="file",
        )
        validation_manifest = _checked_repo_path(
            experiment["validation_manifest"],
            repository,
            label="validation_manifest",
            must_exist=True,
            kind="file",
        )
        test_manifest = _checked_repo_path(
            experiment["test_manifest"],
            repository,
            label="test_manifest",
            must_exist=True,
            kind="file",
        )
        output_root = _checked_repo_path(
            experiment["output_dir"], repository, label="output_dir"
        )
    except (KeyError, AuditPathError) as error:
        raise AuditPathError(f"invalid frozen artifact path: {error}") from error

    data = experiment["data"]
    train_indices = parse_shard_indices(data["train_shards"]) or ()
    validation_indices = parse_shard_indices(data["validation_shards"]) or ()
    test_indices = parse_shard_indices(data["test_shards"]) or ()
    if (
        set(train_indices) & set(validation_indices)
        or set(train_indices) & set(test_indices)
        or set(validation_indices) & set(test_indices)
        or set(train_indices) | set(validation_indices) | set(test_indices) != set(range(100))
    ):
        global_issues.append(
            _issue("data_split_invalid", "train/validation/test shards must partition 0..99")
        )
    if tuple(train_indices) != tuple(range(98)) or tuple(validation_indices) != (98,) or tuple(test_indices) != (99,):
        global_issues.append(
            _issue("data_split_not_frozen", "formal split must be train 000-097, validation 098, test 099")
        )

    manifest_cache: dict[Path, dict[str, Any]] = {}
    for path in {dataset_manifest, validation_manifest}:
        manifest_cache[path] = _audit_manifest(
            path,
            expected_indices=tuple(range(100)),
            expected_seed=int(experiment["task_order_seed"]),
            expected_schema_version=expected_schema_version,
            task_names=task_names,
        )
    training_manifest_audit = manifest_cache[dataset_manifest]
    validation_manifest_audit = manifest_cache[validation_manifest]
    test_manifest_audit = _audit_manifest(
        test_manifest,
        expected_indices=test_indices,
        expected_seed=int(experiment["task_order_seed"]),
        expected_schema_version=expected_schema_version,
        task_names=task_names,
        parent_sha256=training_manifest_audit["sha256"],
        parent_path=dataset_manifest,
        parent_shards=training_manifest_audit["_shards_by_index"],
        split="test",
    )
    manifest_audits = {
        "training": training_manifest_audit,
        "validation": validation_manifest_audit,
        "test": test_manifest_audit,
    }
    manifests_ok = all(value["status"] == "passed" for value in manifest_audits.values())
    if not manifests_ok:
        global_issues.append(_issue("manifest_validation_failed", "one or more manifests failed validation"))

    scan = _scan_output_tree(output_root)
    global_issues.extend(scan["issues"])
    expected_configs = {
        run.run_id: expected_train_config(
            run,
            experiment,
            config_absolute,
            experiment_config_sha256=config_sha256,
        )
        for run in runs
    }
    run_results = [
        audit_run(
            run,
            output_dir=output_root / run.run_id,
            output_root=output_root,
            repository=repository,
            expected_config=expected_configs[run.run_id],
            experiment_config_sha256=config_sha256,
            training_manifest_sha256=training_manifest_audit["sha256"],
            validation_manifest_sha256=validation_manifest_audit["sha256"],
            manifests_ok=manifests_ok,
            validation_tasks=task_names,
            partial_artifacts=scan["partials"],
            symlinks=scan["symlinks"],
        )
        for run in runs
    ]
    passed_count = sum(run["status"] == "passed" for run in run_results)
    incomplete_count = sum(run["status"] == "incomplete" for run in run_results)
    failed_count = sum(run["status"] == "failed" for run in run_results)
    ok = not global_issues and passed_count == len(run_results) == 30
    public_manifest_audits = {
        label: {key: value for key, value in audit.items() if not key.startswith("_")}
        for label, audit in manifest_audits.items()
    }
    summary = {
        "audit_format_version": AUDIT_FORMAT_VERSION,
        "status": "passed" if ok else "failed",
        "ok": ok,
        "protocol_version": experiment.get("protocol_version"),
        "repository": str(repository),
        "config_path": _repo_relative(config_absolute, repository),
        "config_sha256": config_sha256,
        "output_root": _repo_relative(output_root, repository),
        "expected_global_step": int(experiment["training"]["max_steps"]),
        "expected_run_count": 30,
        "run_count": len(run_results),
        "passed_count": passed_count,
        "incomplete_count": incomplete_count,
        "failed_count": failed_count,
        "manifests": public_manifest_audits,
        "partial_artifacts": [
            _display_path(path, output_root) for path in scan["partials"]
        ],
        "symlinks": [_display_path(path, output_root) for path in scan["symlinks"]],
        "issues": global_issues,
        "runs": run_results,
    }
    return _strict_json(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--compact", action="store_true", help="emit JSON on one line")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = audit_experiment(args.config)
    except Exception as error:
        summary = {
            "audit_format_version": AUDIT_FORMAT_VERSION,
            "status": "error",
            "ok": False,
            "config_path": str(args.config),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        exit_code = 2
    else:
        exit_code = 0 if summary["ok"] else 1
    print(
        json.dumps(
            summary,
            indent=None if args.compact else 2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

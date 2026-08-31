"""Plan, audit, and run Henry's nested permutation-task model matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, Iterable

from .generate import V2_SCHEMA_VERSION, task_names_for_schema


DEFAULT_CONFIG = Path("configs/henry_permutation.toml")


def dataset_protocol_version(config: dict[str, Any]) -> str:
    """Return and validate the dataset protocol selected by an experiment."""

    value = config.get("dataset_protocol_version", V2_SCHEMA_VERSION)
    if not isinstance(value, str):
        raise ValueError("dataset_protocol_version must be a string")
    # Validate eagerly so planning cannot silently use an unknown task grid.
    task_names_for_schema(value)
    return value


def task_names_for_experiment(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the schema-aware task registry for an experiment TOML."""

    return task_names_for_schema(dataset_protocol_version(config))


@dataclass(frozen=True)
class ExperimentRun:
    architecture: str
    task_count: int
    tasks: tuple[str, ...]
    seed: int
    run_id: str
    output_dir: str


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    return config, hashlib.sha256(payload).hexdigest()


def build_matrix(config_path: Path = DEFAULT_CONFIG) -> tuple[ExperimentRun, ...]:
    """Build and strictly validate the frozen 30-run experiment matrix."""

    config, _ = _read_config(config_path)
    task_names = task_names_for_experiment(config)
    order = tuple(config["task_order"])
    holdouts = tuple(config["holdout_tasks"])
    subset_sizes = tuple(config["task_subset_sizes"])
    architectures = tuple(config["architectures"])
    seeds = tuple(config["model_seeds"])

    if len(order) != len(task_names) or set(order) != set(task_names):
        raise ValueError("task_order must contain every one of the 20 tasks exactly once")
    if len(holdouts) != 4 or tuple(order[-4:]) != holdouts:
        raise ValueError("holdout_tasks must be exactly the final four shuffled tasks")
    training_pool = order[:-len(holdouts)]
    if subset_sizes != (1, 2, 4, 8, 16):
        raise ValueError("Henry protocol requires nested task sizes 1,2,4,8,16")
    if subset_sizes[-1] != len(training_pool):
        raise ValueError("largest subset must consume the complete non-holdout pool")
    if architectures != ("transformer", "mlp"):
        raise ValueError("architectures must be transformer and mlp")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct model seeds are required for error bars")

    root = Path(config["output_dir"])
    runs: list[ExperimentRun] = []
    for task_count in subset_sizes:
        tasks = training_pool[:task_count]
        for architecture in architectures:
            for seed in seeds:
                run_id = f"{architecture}-tasks{task_count:02d}-seed{seed}"
                runs.append(
                    ExperimentRun(
                        architecture=architecture,
                        task_count=task_count,
                        tasks=tasks,
                        seed=seed,
                        run_id=run_id,
                        output_dir=str(root / run_id),
                    )
                )
    return tuple(runs)


def matrix_summary(
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    runs = build_matrix(config_path)
    dataset_sha256 = _sha256_file(Path(config["dataset_manifest"]))
    validation_sha256 = _sha256_file(Path(config["validation_manifest"]))
    complete = []
    incomplete = []
    for run in runs:
        marker = Path(run.output_dir) / "completed.json"
        target = complete if _completion_is_valid(
            marker,
            experiment_config_sha256=config_sha256,
            expected_steps=int(config["training"]["max_steps"]),
            expected_run=run,
            dataset_sha256=dataset_sha256,
            validation_sha256=validation_sha256,
        ) else incomplete
        target.append(run.run_id)
    return {
        "protocol_version": config["protocol_version"],
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "run_count": len(runs),
        "complete_count": len(complete),
        "incomplete_count": len(incomplete),
        "complete": complete,
        "incomplete": incomplete,
        "runs": [asdict(run) for run in runs],
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completion_is_valid(
    marker_path: Path,
    *,
    experiment_config_sha256: str,
    expected_steps: int,
    expected_run: ExperimentRun | None = None,
    dataset_sha256: str | None = None,
    validation_sha256: str | None = None,
) -> bool:
    """Accept a run only when its marker, config, and checkpoint all agree."""

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return False
        if marker.get("status") != "completed":
            return False
        if marker.get("global_step") != expected_steps:
            return False
        if marker.get("experiment_config_sha256") != experiment_config_sha256:
            return False
        if not isinstance(marker.get("config_sha256"), str):
            return False
        if expected_run is not None:
            if marker.get("run_id") != expected_run.run_id:
                return False
            if marker.get("architecture") != expected_run.architecture:
                return False
            if marker.get("tasks") != list(expected_run.tasks):
                return False
            if marker.get("seed") != expected_run.seed:
                return False
        if (
            dataset_sha256 is not None
            and marker.get("training_manifest_sha256") != dataset_sha256
        ):
            return False
        if (
            validation_sha256 is not None
            and marker.get("validation_manifest_sha256") != validation_sha256
        ):
            return False
        expected_checkpoint_hash = marker.get("checkpoint_sha256")
        if not isinstance(expected_checkpoint_hash, str):
            return False
        checkpoint = Path(marker["checkpoint"])
        if not checkpoint.is_absolute() and not checkpoint.is_file():
            # Training normally records a repo-relative path.  As a fallback,
            # accept a checkpoint beside the marker with the same basename.
            checkpoint = marker_path.parent / checkpoint.name
        return checkpoint.is_file() and _sha256_file(checkpoint) == expected_checkpoint_hash
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        return False


def _training_command(
    run: ExperimentRun,
    config: dict[str, Any],
    config_path: Path,
) -> list[str]:
    training = config["training"]
    model = config["model"]
    data = config["data"]
    return [
        sys.executable,
        "-m",
        "neurips_permutations.training",
        "--manifest",
        str(config["dataset_manifest"]),
        "--validation-manifest",
        str(config["validation_manifest"]),
        "--output-dir",
        run.output_dir,
        "--architecture",
        run.architecture,
        "--tasks",
        ",".join(run.tasks),
        "--seed",
        str(run.seed),
        "--max-steps",
        str(training["max_steps"]),
        "--batch-size",
        str(training["micro_batch_size"]),
        "--grad-accum",
        str(training["gradient_accumulation_steps"]),
        "--max-tokens-per-batch",
        str(training["max_tokens_per_batch"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--warmup-steps",
        str(training["warmup_steps"]),
        "--min-lr-ratio",
        str(training["min_learning_rate_ratio"]),
        "--max-grad-norm",
        str(training["gradient_clip_norm"]),
        "--max-seq-len",
        str(data["max_sequence_length"]),
        "--train-shard-indices",
        str(data["train_shards"]),
        "--validation-shard-indices",
        str(data["validation_shards"]),
        "--shuffle-buffer-size",
        str(data["shuffle_buffer"]),
        "--d-model",
        str(model["d_model"]),
        "--num-layers",
        str(
            model["transformer_layers"]
            if run.architecture == "transformer"
            else model["mlp_layers"]
        ),
        "--num-heads",
        str(model["num_heads"]),
        "--dropout",
        str(model["dropout"]),
        "--mlp-ratio",
        str(model["ff_multiplier"]),
        "--checkpoint-every",
        str(training["checkpoint_every_steps"]),
        "--validate-every",
        str(training["validate_every_steps"]),
        "--validation-batches-per-task",
        str(training["validation_batches_per_task"]),
        "--num-workers",
        str(training["num_workers"]),
        "--experiment-config",
        str(config_path),
        "--resume",
    ]


def run_matrix(
    config_path: Path = DEFAULT_CONFIG,
    *,
    only: Iterable[str] = (),
    dry_run: bool = False,
) -> int:
    """Run every unfinished model sequentially, resuming safely after failure."""

    config, config_sha256 = _read_config(config_path)
    dataset_sha256 = _sha256_file(Path(config["dataset_manifest"]))
    validation_sha256 = _sha256_file(Path(config["validation_manifest"]))
    selected = set(only)
    for run in build_matrix(config_path):
        if selected and run.run_id not in selected:
            continue
        output_dir = Path(run.output_dir)
        if _completion_is_valid(
            output_dir / "completed.json",
            experiment_config_sha256=config_sha256,
            expected_steps=int(config["training"]["max_steps"]),
            expected_run=run,
            dataset_sha256=dataset_sha256,
            validation_sha256=validation_sha256,
        ):
            continue
        command = _training_command(run, config, config_path)
        if dry_run:
            print(json.dumps({"run_id": run.run_id, "command": command}))
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return result.returncode
        if not _completion_is_valid(
            output_dir / "completed.json",
            experiment_config_sha256=config_sha256,
            expected_steps=int(config["training"]["max_steps"]),
            expected_run=run,
            dataset_sha256=dataset_sha256,
            validation_sha256=validation_sha256,
        ):
            raise RuntimeError(f"training exited without completion marker: {run.run_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan or args.status:
        print(json.dumps(matrix_summary(args.config), indent=2))
        return 0
    return run_matrix(args.config, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

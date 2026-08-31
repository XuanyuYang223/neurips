"""Plan, inspect, and run Henry's permutation-task model matrices."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any, Iterable, Literal

from .generate import V2_SCHEMA_VERSION, task_names_for_schema


DEFAULT_CONFIG = Path("configs/henry_permutation.toml")
CATEGORY_OUTPUT_SUBDIR = "category-comparison"
MatrixName = Literal["nested", "category"]

_CATEGORY_TASKS = {
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
_CATEGORY_BATCH_OVERRIDES = {
    "encoding_e4": (4, 16),
    "statistics_s4": (16, 4),
    "algebra_a4": (16, 4),
}
_CONDITION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCALING_CONDITIONS: dict[str, dict[str, Any]] = {
    "data10x_model1x": {
        "data_multiplier": 10,
        "model_multiplier": 1,
        "training_examples_per_run": 12_800_000,
        "dataset_manifest": "data/permutation-100m-v3-scaling/manifest.json",
        "training_manifest_sha256": "6bafe42be4adc2fd956275af171d9efece40357c6de9cc5791b5514bda34591f",
        "training_parent_records": 100_000_000,
        "training_used_records": 98_000_000,
        "output_dir": "runs/permutation-scaling-v3/data10x-model1x",
        "max_steps": 200_000,
        "warmup_steps": 10_000,
        "interval": 10_000,
        "transformer_layers": 4,
        "mlp_layers": 1,
        "train_shards": "000-979",
    },
    "data1x_model2x": {
        "data_multiplier": 1,
        "model_multiplier": 2,
        "training_examples_per_run": 1_280_000,
        "dataset_manifest": "data/permutation-10m-v3/manifest.json",
        "training_manifest_sha256": "b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f",
        "training_parent_records": 10_000_000,
        "training_used_records": 9_800_000,
        "output_dir": "runs/permutation-scaling-v3/data1x-model2x",
        "max_steps": 20_000,
        "warmup_steps": 1_000,
        "interval": 1_000,
        "transformer_layers": 8,
        "mlp_layers": 2,
        "train_shards": "000-097",
    },
    "data10x_model2x": {
        "data_multiplier": 10,
        "model_multiplier": 2,
        "training_examples_per_run": 12_800_000,
        "dataset_manifest": "data/permutation-100m-v3-scaling/manifest.json",
        "training_manifest_sha256": "6bafe42be4adc2fd956275af171d9efece40357c6de9cc5791b5514bda34591f",
        "training_parent_records": 100_000_000,
        "training_used_records": 98_000_000,
        "output_dir": "runs/permutation-scaling-v3/data10x-model2x",
        "max_steps": 200_000,
        "warmup_steps": 10_000,
        "interval": 10_000,
        "transformer_layers": 8,
        "mlp_layers": 2,
        "train_shards": "000-979",
    },
}


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


@dataclass(frozen=True)
class CategoryExperimentRun(ExperimentRun):
    """One task-count-matched E4, S4, or A4 comparison run."""

    condition: str
    micro_batch_size: int
    gradient_accumulation_steps: int


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    return config, hashlib.sha256(payload).hexdigest()


def _validate_optional_scaling_config(config: dict[str, Any]) -> None:
    """Freeze the three non-baseline cells of the data/capacity study."""

    scaling = config.get("scaling")
    if scaling is None:
        return
    if not isinstance(scaling, dict) or set(scaling) != {
        "condition",
        "data_multiplier",
        "model_multiplier",
        "training_examples_per_run",
        "baseline_config",
    }:
        raise ValueError("scaling must contain the complete frozen condition schema")
    condition = scaling.get("condition")
    expected = _SCALING_CONDITIONS.get(condition)
    if expected is None:
        raise ValueError(f"unknown scaling condition: {condition!r}")
    for key in (
        "data_multiplier",
        "model_multiplier",
        "training_examples_per_run",
    ):
        if scaling.get(key) != expected[key]:
            raise ValueError(f"scaling {condition} has invalid {key}")
    if scaling.get("baseline_config") != "configs/henry_permutation_revised.toml":
        raise ValueError("scaling baseline_config must identify the completed v3 baseline")
    for key in ("dataset_manifest", "output_dir"):
        if config.get(key) != expected[key]:
            raise ValueError(f"scaling {condition} has invalid {key}")
    if config.get("validation_manifest") != "data/permutation-10m-v3/manifest.json":
        raise ValueError("scaling validation must use the frozen v3 parent manifest")
    if config.get("test_manifest") != "data/permutation-10m-v3/test_manifest.json":
        raise ValueError("scaling test must use the frozen v3 test manifest")
    artifact = config.get("dataset_artifact")
    expected_artifact = {
        "training_manifest_sha256": expected["training_manifest_sha256"],
        "validation_manifest_sha256": "b20a16cee7710cee4a21cc4575c8651ade1bcfca18219d2e6c230d4a3ab0cf6f",
        "test_manifest_sha256": "3ca12e6b6eeb29fc0ddd441b9c44c80d7a160faaf7e832eb55007f4c6a3ab52b",
        "training_parent_records": expected["training_parent_records"],
        "training_used_records": expected["training_used_records"],
    }
    if artifact != expected_artifact:
        raise ValueError("scaling dataset_artifact does not match the frozen manifests")

    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    if not all(isinstance(value, dict) for value in (data, model, training)):
        raise ValueError("scaling config requires data, model, and training tables")
    assert isinstance(data, dict) and isinstance(model, dict) and isinstance(training, dict)
    checks = {
        "train_shards": (data.get("train_shards"), expected["train_shards"]),
        "validation_shards": (data.get("validation_shards"), "098"),
        "test_shards": (data.get("test_shards"), "099"),
        "transformer_layers": (
            model.get("transformer_layers"),
            expected["transformer_layers"],
        ),
        "mlp_layers": (model.get("mlp_layers"), expected["mlp_layers"]),
        "max_steps": (training.get("max_steps"), expected["max_steps"]),
        "warmup_steps": (training.get("warmup_steps"), expected["warmup_steps"]),
        "checkpoint_every_steps": (
            training.get("checkpoint_every_steps"),
            expected["interval"],
        ),
        "validate_every_steps": (
            training.get("validate_every_steps"),
            expected["interval"],
        ),
        "micro_batch_size": (training.get("micro_batch_size"), 16),
        "gradient_accumulation_steps": (
            training.get("gradient_accumulation_steps"),
            4,
        ),
    }
    for label, (actual, wanted) in checks.items():
        if actual != wanted:
            raise ValueError(
                f"scaling {condition} requires {label}={wanted!r}, found {actual!r}"
            )

    sources = config.get("data_sources")
    if expected["data_multiplier"] == 10:
        wanted_sources = {
            "training_manifest_shard_count": 1_000,
            "training_manifest_seed": 20_260_831,
            "validation_manifest_shard_count": 100,
            "validation_manifest_seed": 20_260_830,
            "test_parent": "validation",
        }
        if sources != wanted_sources:
            raise ValueError("10x scaling must use the frozen cross-manifest data sources")
    elif sources is not None:
        raise ValueError("1x scaling must use the original single-manifest data split")


def build_matrix(config_path: Path = DEFAULT_CONFIG) -> tuple[ExperimentRun, ...]:
    """Build and strictly validate the frozen 30-run experiment matrix."""

    config, _ = _read_config(config_path)
    _validate_optional_scaling_config(config)
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


def build_category_matrix(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[CategoryExperimentRun, ...]:
    """Build and strictly validate the frozen 18-run E4/S4/A4 matrix."""

    config, _ = _read_config(config_path)
    task_names = set(task_names_for_experiment(config))
    comparison = config.get("category_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("category_comparison must be a TOML table")

    architectures = tuple(comparison.get("architectures", ()))
    seeds = tuple(comparison.get("model_seeds", ()))
    if architectures != ("transformer", "mlp"):
        raise ValueError(
            "category_comparison architectures must be transformer and mlp"
        )
    if architectures != tuple(config.get("architectures", ())):
        raise ValueError(
            "category_comparison architectures must match the nested matrix"
        )
    if len(seeds) != 3 or len(set(seeds)) != 3 or not all(
        type(seed) is int for seed in seeds
    ):
        raise ValueError(
            "category_comparison requires exactly three distinct integer model seeds"
        )
    if seeds != tuple(config.get("model_seeds", ())):
        raise ValueError(
            "category_comparison model_seeds must match the nested matrix"
        )
    if comparison.get("records_per_task") != 500_000:
        raise ValueError(
            "category_comparison records_per_task must be the frozen 500000"
        )
    if comparison.get("equal_optimizer_updates") is not True:
        raise ValueError(
            "category_comparison must keep equal_optimizer_updates enabled"
        )

    raw_conditions = comparison.get("conditions")
    if not isinstance(raw_conditions, list):
        raise ValueError("category_comparison.conditions must be an array of tables")
    if len(raw_conditions) != len(_CATEGORY_TASKS):
        raise ValueError("category_comparison must define exactly E4, S4, and A4")

    conditions: list[tuple[str, tuple[str, ...], int, int]] = []
    used_tasks: set[str] = set()
    for position, raw in enumerate(raw_conditions):
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "tasks",
            "micro_batch_size",
            "gradient_accumulation_steps",
        }:
            raise ValueError(
                "each category_comparison condition must contain name, tasks, "
                "micro_batch_size, and gradient_accumulation_steps"
            )
        name = raw["name"]
        tasks = raw["tasks"]
        if not isinstance(name, str) or not _CONDITION_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid category condition name at index {position}")
        if not isinstance(tasks, list) or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(f"category condition {name} tasks must be strings")
        task_tuple = tuple(tasks)
        expected_tasks = _CATEGORY_TASKS.get(name)
        if expected_tasks is None:
            raise ValueError(f"unknown category condition: {name}")
        if task_tuple != expected_tasks:
            raise ValueError(
                f"category condition {name} must use its frozen four-task set"
            )
        if len(set(task_tuple)) != 4 or not set(task_tuple) <= task_names:
            raise ValueError(
                f"category condition {name} must contain four unique dataset tasks"
            )
        overlap = used_tasks.intersection(task_tuple)
        if overlap:
            raise ValueError(
                "category conditions must be task-disjoint; overlap: "
                + ",".join(sorted(overlap))
            )
        used_tasks.update(task_tuple)
        micro_batch_size = raw["micro_batch_size"]
        gradient_accumulation_steps = raw["gradient_accumulation_steps"]
        if (
            type(micro_batch_size) is not int
            or micro_batch_size < 1
            or type(gradient_accumulation_steps) is not int
            or gradient_accumulation_steps < 1
        ):
            raise ValueError(
                f"category condition {name} batch override values must be positive integers"
            )
        if micro_batch_size * gradient_accumulation_steps != 64:
            raise ValueError(
                f"category condition {name} must have effective batch size 64"
            )
        expected_batch_override = _CATEGORY_BATCH_OVERRIDES[name]
        if (micro_batch_size, gradient_accumulation_steps) != expected_batch_override:
            raise ValueError(
                f"category condition {name} must use frozen batch override "
                f"{expected_batch_override[0]}x{expected_batch_override[1]}"
            )
        conditions.append(
            (
                name,
                task_tuple,
                micro_batch_size,
                gradient_accumulation_steps,
            )
        )

    if tuple(name for name, *_ in conditions) != tuple(_CATEGORY_TASKS):
        raise ValueError("category conditions must be ordered E4, S4, then A4")

    root = Path(config["output_dir"]) / CATEGORY_OUTPUT_SUBDIR
    runs: list[CategoryExperimentRun] = []
    for condition, tasks, micro_batch_size, gradient_accumulation_steps in conditions:
        condition_id = condition.replace("_", "-")
        for architecture in architectures:
            for seed in seeds:
                run_id = f"category-{condition_id}-{architecture}-seed{seed}"
                runs.append(
                    CategoryExperimentRun(
                        architecture=architecture,
                        task_count=4,
                        tasks=tasks,
                        seed=seed,
                        run_id=run_id,
                        output_dir=str(root / run_id),
                        condition=condition,
                        micro_batch_size=micro_batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )
                )
    if len(runs) != 18 or len({run.run_id for run in runs}) != 18:
        raise ValueError("category matrix must contain exactly 18 unique runs")
    return tuple(runs)


def build_experiment_matrix(
    config_path: Path = DEFAULT_CONFIG,
    *,
    matrix: MatrixName = "nested",
) -> tuple[ExperimentRun, ...]:
    """Dispatch to a frozen matrix while preserving the legacy nested API."""

    if matrix == "nested":
        return build_matrix(config_path)
    if matrix == "category":
        return build_category_matrix(config_path)
    raise ValueError(f"unknown experiment matrix: {matrix}")


def matrix_summary(
    config_path: Path = DEFAULT_CONFIG,
    *,
    matrix: MatrixName = "nested",
) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    runs = build_experiment_matrix(config_path, matrix=matrix)
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
        "matrix": matrix,
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
    micro_batch_size = (
        run.micro_batch_size
        if isinstance(run, CategoryExperimentRun)
        else training["micro_batch_size"]
    )
    gradient_accumulation_steps = (
        run.gradient_accumulation_steps
        if isinstance(run, CategoryExperimentRun)
        else training["gradient_accumulation_steps"]
    )
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
        str(micro_batch_size),
        "--grad-accum",
        str(gradient_accumulation_steps),
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
    matrix: MatrixName = "nested",
    only: Iterable[str] = (),
    dry_run: bool = False,
) -> int:
    """Run every unfinished model sequentially, resuming safely after failure."""

    config, config_sha256 = _read_config(config_path)
    dataset_sha256 = _sha256_file(Path(config["dataset_manifest"]))
    validation_sha256 = _sha256_file(Path(config["validation_manifest"]))
    selected = set(only)
    for run in build_experiment_matrix(config_path, matrix=matrix):
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
    parser.add_argument(
        "--matrix",
        choices=("nested", "category"),
        default="nested",
        help="select the 30-run nested or 18-run E4/S4/A4 matrix",
    )
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
        print(json.dumps(matrix_summary(args.config, matrix=args.matrix), indent=2))
        return 0
    return run_matrix(
        args.config,
        matrix=args.matrix,
        only=args.only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

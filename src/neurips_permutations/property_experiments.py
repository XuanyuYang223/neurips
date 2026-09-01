"""Plan and run the frozen 32-property zero-overlap pilot matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, Sequence

from .generate import PROPERTY32_SCHEMA_VERSION, task_names_for_schema


DEFAULT_CONFIG = Path("configs/property32_zero_overlap_pilot.toml")


@dataclass(frozen=True)
class PropertyExperimentRun:
    pool: str
    architecture: str
    task_count: int
    tasks: tuple[str, ...]
    seed: int
    run_id: str
    output_dir: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = tomllib.loads(payload.decode("utf-8"))
    return value, hashlib.sha256(payload).hexdigest()


def build_property_matrix(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[PropertyExperimentRun, ...]:
    """Return the strictly validated A/B zero-overlap pilot runs."""

    config, _ = _read_config(config_path)
    if config.get("dataset_protocol_version") != PROPERTY32_SCHEMA_VERSION:
        raise ValueError("property pilot must use the frozen property32 schema")
    registry = set(task_names_for_schema(PROPERTY32_SCHEMA_VERSION))
    pool_a = tuple(config.get("pool_a", ()))
    pool_b = tuple(config.get("pool_b", ()))
    if len(pool_a) != 16 or len(set(pool_a)) != 16:
        raise ValueError("pool_a must contain 16 unique properties")
    if len(pool_b) != 16 or len(set(pool_b)) != 16:
        raise ValueError("pool_b must contain 16 unique properties")
    if set(pool_a) & set(pool_b) or set(pool_a) | set(pool_b) != registry:
        raise ValueError("pool_a and pool_b must partition all 32 properties")
    subset_sizes = tuple(config.get("task_subset_sizes", ()))
    if subset_sizes != (1, 2, 4, 8, 16):
        raise ValueError("property pilot requires k = 1, 2, 4, 8, 16")
    architectures = tuple(config.get("architectures", ()))
    if not architectures or not set(architectures) <= {"transformer", "mlp"}:
        raise ValueError("architectures must contain transformer and/or mlp")
    seeds = tuple(config.get("model_seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds) or not all(
        type(seed) is int for seed in seeds
    ):
        raise ValueError("model_seeds must be distinct integers")

    artifact = config.get("dataset_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("dataset_artifact must be a TOML table")
    for config_key, artifact_key in (
        ("dataset_manifest", "train_manifest_sha256"),
        ("validation_manifest", "validation_manifest_sha256"),
        ("test_manifest", "test_manifest_sha256"),
    ):
        path = Path(config[config_key])
        if _sha256(path) != artifact.get(artifact_key):
            raise ValueError(f"{config_key} does not match its frozen SHA-256")

    root = Path(config["output_dir"])
    runs: list[PropertyExperimentRun] = []
    for pool_name, pool_tasks in (("a", pool_a), ("b", pool_b)):
        for task_count in subset_sizes:
            tasks = pool_tasks[:task_count]
            for architecture in architectures:
                for seed in seeds:
                    run_id = (
                        f"property-{pool_name}-{architecture}-"
                        f"tasks{task_count:02d}-seed{seed}"
                    )
                    runs.append(
                        PropertyExperimentRun(
                            pool=pool_name,
                            architecture=architecture,
                            task_count=task_count,
                            tasks=tasks,
                            seed=seed,
                            run_id=run_id,
                            output_dir=str(root / run_id),
                        )
                    )
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("property pilot run IDs are not unique")
    return tuple(runs)


def _training_command(
    run: PropertyExperimentRun,
    config: dict[str, Any],
    config_path: Path,
) -> list[str]:
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
        "--experiment-config",
        str(config_path),
        "--resume",
    ]


def _completed(run: PropertyExperimentRun, config_sha256: str, steps: int) -> bool:
    marker_path = Path(run.output_dir) / "completed.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = Path(marker["checkpoint"])
        if not checkpoint.is_file():
            checkpoint = marker_path.parent / checkpoint.name
        return (
            marker.get("status") == "completed"
            and marker.get("global_step") == steps
            and marker.get("run_id") == run.run_id
            and marker.get("architecture") == run.architecture
            and marker.get("tasks") == list(run.tasks)
            and marker.get("seed") == run.seed
            and marker.get("experiment_config_sha256") == config_sha256
            and checkpoint.is_file()
            and marker.get("checkpoint_sha256") == _sha256(checkpoint)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def matrix_summary(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    runs = build_property_matrix(config_path)
    steps = int(config["training"]["max_steps"])
    completed = [run.run_id for run in runs if _completed(run, config_sha256, steps)]
    return {
        "protocol_version": config["protocol_version"],
        "config_sha256": config_sha256,
        "run_count": len(runs),
        "complete_count": len(completed),
        "incomplete_count": len(runs) - len(completed),
        "complete": completed,
        "runs": [asdict(run) for run in runs],
    }


def run_matrix(config_path: Path = DEFAULT_CONFIG, *, dry_run: bool = False) -> None:
    config, config_sha256 = _read_config(config_path)
    runs = build_property_matrix(config_path)
    steps = int(config["training"]["max_steps"])
    for run in runs:
        if _completed(run, config_sha256, steps):
            continue
        command = _training_command(run, config, config_path)
        if dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan or args.status:
        print(json.dumps(matrix_summary(args.config), indent=2, sort_keys=True))
    else:
        run_matrix(args.config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "PropertyExperimentRun",
    "build_parser",
    "build_property_matrix",
    "main",
    "matrix_summary",
    "run_matrix",
]

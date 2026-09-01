"""Plan, run, and inspect the frozen relation-controlled property matrix."""

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

from .generate import PROPERTY32_SCHEMA_VERSION
from .math_ops import PROPERTY32_TASK_NAMES
from .property_replicates import NATURAL_DUAL_PAIRS, PROPERTY_FAMILIES


DEFAULT_CONFIG = Path("configs/property32_relation_controlled.toml")
TASK_COUNTS = (1, 2, 4, 8)
SPLIT_IDS = ("s0", "s1", "s2")
MODEL_SEEDS = (17, 42, 101)


@dataclass(frozen=True)
class RelationExperimentRun:
    split_id: str
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
    return tomllib.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def _validate_pool_order(pool: tuple[str, ...], *, split_id: str, side: str) -> None:
    if len(pool) != 8 or len(set(pool)) != 8:
        raise ValueError(f"{split_id} pool {side} must contain eight unique tasks")
    for block_start in (0, 4):
        block = set(pool[block_start : block_start + 4])
        if [len(block & family) for family in PROPERTY_FAMILIES] != [1, 1, 1, 1]:
            raise ValueError(
                f"{split_id} pool {side} block {block_start // 4} is not family-balanced"
            )


def validate_relation_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(path)
    if config.get("protocol_version") != "property32-relation-controlled/v1":
        raise ValueError("unexpected relation-controlled protocol version")
    if config.get("dataset_protocol_version") != PROPERTY32_SCHEMA_VERSION:
        raise ValueError("relation-controlled study requires the property32 corpus")
    if tuple(config.get("task_subset_sizes", ())) != TASK_COUNTS:
        raise ValueError("relation-controlled study requires k = 1, 2, 4, 8")
    if tuple(config.get("model_seeds", ())) != MODEL_SEEDS:
        raise ValueError("relation-controlled model seeds drifted")
    if tuple(config.get("split_ids", ())) != SPLIT_IDS:
        raise ValueError("relation-controlled split IDs drifted")
    if tuple(config.get("architectures", ())) != ("transformer",):
        raise ValueError("relation-controlled study is Transformer-only")

    artifact = config.get("dataset_artifact")
    if not isinstance(artifact, dict):
        raise ValueError("dataset_artifact must be a table")
    for key, hash_key in (
        ("dataset_manifest", "train_manifest_sha256"),
        ("validation_manifest", "validation_manifest_sha256"),
        ("test_manifest", "test_manifest_sha256"),
    ):
        if _sha256(Path(config[key])) != artifact.get(hash_key):
            raise ValueError(f"{key} does not match its frozen SHA-256")

    registry = set(PROPERTY32_TASK_NAMES)
    split_summary: dict[str, Any] = {}
    for split_id in SPLIT_IDS:
        value = config["splits"][split_id]
        pool_a = tuple(value["pool_a"])
        pool_b = tuple(value["pool_b"])
        _validate_pool_order(pool_a, split_id=split_id, side="a")
        _validate_pool_order(pool_b, split_id=split_id, side="b")
        selected = set(pool_a) | set(pool_b)
        if set(pool_a) & set(pool_b) or len(selected) != 16 or not selected <= registry:
            raise ValueError(f"{split_id} pools must be disjoint and select 16 properties")
        dual_counts = [int(left in selected) + int(right in selected) for left, right in NATURAL_DUAL_PAIRS]
        if dual_counts != [1] * len(NATURAL_DUAL_PAIRS):
            raise ValueError(f"{split_id} must select exactly one member of every natural dual")
        correlations = {
            name: float(value[f"conditional_cross_{name}_abs_correlation"])
            for name in ("mean", "q95", "max")
        }
        if not (0 <= correlations["mean"] <= correlations["q95"] <= correlations["max"] < 0.4):
            raise ValueError(f"{split_id} correlation controls are outside the frozen bounds")
        split_summary[split_id] = {
            "pool_a": pool_a,
            "pool_b": pool_b,
            "selected_tasks": len(selected),
            "co_selected_natural_duals": 0,
            "conditional_cross_correlations": correlations,
        }
    return {
        "config_sha256": config_sha256,
        "split_count": len(SPLIT_IDS),
        "seed_count": len(MODEL_SEEDS),
        "cell_count": len(SPLIT_IDS) * len(MODEL_SEEDS),
        "run_count": len(SPLIT_IDS) * len(MODEL_SEEDS) * 2 * len(TASK_COUNTS),
        "splits": split_summary,
    }


def build_relation_matrix(
    path: Path = DEFAULT_CONFIG,
) -> tuple[RelationExperimentRun, ...]:
    validate_relation_config(path)
    config, _ = _read_config(path)
    root = Path(config["output_dir"])
    runs: list[RelationExperimentRun] = []
    for split_id in SPLIT_IDS:
        split = config["splits"][split_id]
        for seed in MODEL_SEEDS:
            cell_root = root / f"{split_id}-seed{seed}"
            for pool, tasks in (("a", tuple(split["pool_a"])), ("b", tuple(split["pool_b"]))):
                for task_count in TASK_COUNTS:
                    run_id = (
                        f"relation-{split_id}-{pool}-transformer-"
                        f"tasks{task_count:02d}-seed{seed}"
                    )
                    runs.append(
                        RelationExperimentRun(
                            split_id=split_id,
                            pool=pool,
                            architecture="transformer",
                            task_count=task_count,
                            tasks=tasks[:task_count],
                            seed=seed,
                            run_id=run_id,
                            output_dir=str(cell_root / run_id),
                        )
                    )
    if len(runs) != 72 or len({run.run_id for run in runs}) != 72:
        raise ValueError("relation-controlled matrix must contain 72 unique runs")
    return tuple(runs)


def _completed(
    run: RelationExperimentRun, *, config_sha256: str, steps: int
) -> bool:
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


def relation_summary(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(path)
    design = validate_relation_config(path)
    runs = build_relation_matrix(path)
    steps = int(config["training"]["max_steps"])
    completed = [
        run.run_id
        for run in runs
        if _completed(run, config_sha256=config_sha256, steps=steps)
    ]
    cells: dict[str, dict[str, int]] = {}
    for split_id in SPLIT_IDS:
        for seed in MODEL_SEEDS:
            key = f"{split_id}:{seed}"
            selected = [run for run in runs if run.split_id == split_id and run.seed == seed]
            cells[key] = {
                "run_count": len(selected),
                "complete_count": sum(run.run_id in completed for run in selected),
            }
    return {
        "protocol_version": config["protocol_version"],
        "config_sha256": config_sha256,
        "run_count": len(runs),
        "complete_count": len(completed),
        "incomplete_count": len(runs) - len(completed),
        "design": design,
        "cells": cells,
        "complete": completed,
        "runs": [asdict(run) for run in runs],
    }


def _training_command(
    run: RelationExperimentRun, config: dict[str, Any], config_path: Path
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


def run_relation_matrix(
    path: Path = DEFAULT_CONFIG,
    *,
    cell: str = "all",
    dry_run: bool = False,
) -> None:
    config, config_sha256 = _read_config(path)
    runs = build_relation_matrix(path)
    valid_cells = {f"{split_id}:{seed}" for split_id in SPLIT_IDS for seed in MODEL_SEEDS}
    if cell != "all" and cell not in valid_cells:
        raise ValueError(f"unknown relation-controlled cell {cell!r}")
    steps = int(config["training"]["max_steps"])
    for run in runs:
        run_cell = f"{run.split_id}:{run.seed}"
        if cell != "all" and run_cell != cell:
            continue
        if _completed(run, config_sha256=config_sha256, steps=steps):
            continue
        command = _training_command(run, config, path)
        if dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cell", default="all")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan or args.status:
        print(json.dumps(relation_summary(args.config), indent=2, sort_keys=True))
    else:
        run_relation_matrix(args.config, cell=args.cell, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "MODEL_SEEDS",
    "RelationExperimentRun",
    "SPLIT_IDS",
    "TASK_COUNTS",
    "build_relation_matrix",
    "main",
    "relation_summary",
    "run_relation_matrix",
    "validate_relation_config",
]

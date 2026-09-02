"""Plan, run, and inspect the frozen combinatorial task-geometry matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, Mapping, Sequence

from .generate import PROPERTY32_SCHEMA_VERSION
from .math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS, inverse


DEFAULT_CONFIG = Path("configs/property_task_geometry.toml")
MODEL_SEEDS = (17, 42, 101)
BUNDLE_SPLIT_IDS = ("s0", "s1", "s2", "s3")
RELATED_PAIR_COUNTS = (0, 1, 2, 4)


@dataclass(frozen=True)
class TaskRelation:
    pair_id: str
    left: str
    right: str
    input_transform: str
    right_label_offset: int


@dataclass(frozen=True)
class GeometryRun:
    kind: str
    split_id: str | None
    role: str
    related_pair_count: int | None
    architecture: str
    tasks: tuple[str, ...]
    seed: int
    run_id: str
    canonical_run_id: str
    output_dir: str
    experiment_config: str
    reused: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return tomllib.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def relation_definitions(path: Path = DEFAULT_CONFIG) -> tuple[TaskRelation, ...]:
    config, _ = _read_config(path)
    values = config.get("relation_pairs")
    if not isinstance(values, list):
        raise ValueError("relation_pairs must be an array of tables")
    try:
        return tuple(
            TaskRelation(
                pair_id=str(value["pair_id"]),
                left=str(value["left"]),
                right=str(value["right"]),
                input_transform=str(value["input_transform"]),
                right_label_offset=int(value["right_label_offset"]),
            )
            for value in values
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid relation-pair table") from error


def transform_permutation(
    permutation: Sequence[int], transformation: str
) -> tuple[int, ...]:
    values = tuple(permutation)
    if transformation == "identity":
        return values
    if transformation == "inverse":
        return inverse(values)
    if transformation == "complement":
        size = len(values)
        return tuple(size + 1 - value for value in values)
    raise ValueError(f"unknown permutation transformation {transformation!r}")


def verify_relation_identities(
    path: Path = DEFAULT_CONFIG, *, max_n: int = 8
) -> dict[str, Any]:
    if not 2 <= max_n <= 9:
        raise ValueError("max_n must be between 2 and 9")
    relations = relation_definitions(path)
    checked = 0
    for size in range(2, max_n + 1):
        for value in permutations(range(1, size + 1)):
            checked += 1
            for relation in relations:
                transformed = transform_permutation(value, relation.input_transform)
                left = PROPERTY_FUNCTIONS[relation.left](value)
                right = PROPERTY_FUNCTIONS[relation.right](transformed)
                if right != left + relation.right_label_offset:
                    raise ValueError(
                        f"relation {relation.pair_id} failed on {value}: "
                        f"left={left}, right={right}"
                    )
    return {
        "max_n": max_n,
        "permutations_checked": checked,
        "relations_checked": len(relations),
    }


def _direct_relation_count(
    left: Sequence[str], right: Sequence[str], task_to_pair: Mapping[str, str]
) -> int:
    return len({task_to_pair[task] for task in left} & {task_to_pair[task] for task in right})


def validate_geometry_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(path)
    if config.get("protocol_version") != "property-task-geometry/v1":
        raise ValueError("unexpected task-geometry protocol version")
    if config.get("dataset_protocol_version") != PROPERTY32_SCHEMA_VERSION:
        raise ValueError("task geometry requires the property32 corpus")
    if tuple(config.get("model_seeds", ())) != MODEL_SEEDS:
        raise ValueError("task-geometry model seeds drifted")
    if tuple(config.get("bundle_split_ids", ())) != BUNDLE_SPLIT_IDS:
        raise ValueError("task-geometry split IDs drifted")
    if tuple(config.get("related_pair_counts", ())) != RELATED_PAIR_COUNTS:
        raise ValueError("task-geometry relation-count conditions drifted")

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

    relations = relation_definitions(path)
    if len(relations) != 8 or len({relation.pair_id for relation in relations}) != 8:
        raise ValueError("task geometry requires eight uniquely named relations")
    selected_tasks = tuple(
        task for relation in relations for task in (relation.left, relation.right)
    )
    if len(selected_tasks) != 16 or len(set(selected_tasks)) != 16:
        raise ValueError("relation endpoints must be sixteen unique tasks")
    if not set(selected_tasks) <= set(PROPERTY32_TASK_NAMES):
        raise ValueError("relation endpoints must belong to the property32 registry")
    if any(
        relation.input_transform not in {"inverse", "complement"}
        or relation.right_label_offset not in {0, 1}
        for relation in relations
    ):
        raise ValueError("relation transformations or offsets drifted")
    task_to_pair = {
        task: relation.pair_id
        for relation in relations
        for task in (relation.left, relation.right)
    }

    bundles = config.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != 4:
        raise ValueError("task geometry requires four bundle layouts")
    if tuple(bundle.get("split_id") for bundle in bundles) != BUNDLE_SPLIT_IDS:
        raise ValueError("bundle layout order drifted")
    anchor_counts: Counter[str] = Counter()
    condition_counts = {count: Counter() for count in RELATED_PAIR_COUNTS}
    bundle_sets: list[frozenset[str]] = []
    for bundle in bundles:
        anchor = tuple(bundle.get("anchor", ()))
        if len(anchor) != 4 or len(set(anchor)) != 4 or not set(anchor) <= set(selected_tasks):
            raise ValueError("every anchor must contain four selected tasks")
        anchor_counts.update(anchor)
        bundle_sets.append(frozenset(anchor))
        for count in RELATED_PAIR_COUNTS:
            tasks = tuple(bundle.get(f"b_r{count}", ()))
            if len(tasks) != 4 or len(set(tasks)) != 4 or not set(tasks) <= set(selected_tasks):
                raise ValueError(f"every r={count} bundle must contain four selected tasks")
            if set(anchor) & set(tasks):
                raise ValueError("paired bundles cannot share a task name")
            observed = _direct_relation_count(anchor, tasks, task_to_pair)
            if observed != count:
                raise ValueError(
                    f"{bundle['split_id']} r={count} has {observed} direct relations"
                )
            condition_counts[count].update(tasks)
            bundle_sets.append(frozenset(tasks))
    expected_counts = Counter({task: 1 for task in selected_tasks})
    if anchor_counts != expected_counts:
        raise ValueError("anchor layouts must use every task exactly once")
    if any(counts != expected_counts for counts in condition_counts.values()):
        raise ValueError("every r condition must use every task exactly once")
    if len(bundle_sets) != len(set(bundle_sets)):
        raise ValueError("four-task bundle designs must be unique")

    reuse = config.get("checkpoint_reuse")
    if not isinstance(reuse, list) or len(reuse) != 8:
        raise ValueError("the frozen protocol requires eight reuse entries")
    reuse_keys: set[tuple[str, int]] = set()
    for value in reuse:
        key = (str(value.get("task")), int(value.get("seed", -1)))
        if key in reuse_keys or key[0] not in selected_tasks or key[1] not in MODEL_SEEDS:
            raise ValueError("invalid or duplicate checkpoint-reuse entry")
        reuse_keys.add(key)
        if not Path(str(value.get("experiment_config"))).is_file():
            raise ValueError("checkpoint-reuse experiment config is missing")

    return {
        "protocol_version": config["protocol_version"],
        "config_sha256": config_sha256,
        "selected_tasks": selected_tasks,
        "relation_count": len(relations),
        "specialist_run_count": len(selected_tasks) * len(MODEL_SEEDS),
        "bundle_run_count": len(bundles) * (1 + len(RELATED_PAIR_COUNTS)) * len(MODEL_SEEDS),
        "run_count": 108,
        "reuse_count": len(reuse),
    }


def build_geometry_matrix(path: Path = DEFAULT_CONFIG) -> tuple[GeometryRun, ...]:
    design = validate_geometry_config(path)
    config, _ = _read_config(path)
    root = Path(config["output_dir"])
    reuse = {
        (str(value["task"]), int(value["seed"])): value
        for value in config["checkpoint_reuse"]
    }
    runs: list[GeometryRun] = []
    for task in design["selected_tasks"]:
        for seed in MODEL_SEEDS:
            run_id = f"geometry-specialist-{task}-transformer-seed{seed}"
            source = reuse.get((task, seed))
            if source is None:
                canonical_run_id = run_id
                output_dir = str(root / "specialists" / run_id)
                experiment_config = str(path)
                reused = False
            else:
                canonical_run_id = str(source["run_id"])
                output_dir = str(source["output_dir"])
                experiment_config = str(source["experiment_config"])
                reused = True
            runs.append(
                GeometryRun(
                    kind="specialist",
                    split_id=None,
                    role="specialist",
                    related_pair_count=None,
                    architecture="transformer",
                    tasks=(task,),
                    seed=seed,
                    run_id=run_id,
                    canonical_run_id=canonical_run_id,
                    output_dir=output_dir,
                    experiment_config=experiment_config,
                    reused=reused,
                )
            )
    for bundle in config["bundles"]:
        split_id = str(bundle["split_id"])
        roles: list[tuple[str, int | None, tuple[str, ...]]] = [
            ("anchor", None, tuple(bundle["anchor"]))
        ]
        roles.extend(
            (f"r{count}", count, tuple(bundle[f"b_r{count}"]))
            for count in RELATED_PAIR_COUNTS
        )
        for role, count, tasks in roles:
            for seed in MODEL_SEEDS:
                run_id = f"geometry-bundle-{split_id}-{role}-transformer-seed{seed}"
                runs.append(
                    GeometryRun(
                        kind="bundle",
                        split_id=split_id,
                        role=role,
                        related_pair_count=count,
                        architecture="transformer",
                        tasks=tasks,
                        seed=seed,
                        run_id=run_id,
                        canonical_run_id=run_id,
                        output_dir=str(root / "bundles" / split_id / run_id),
                        experiment_config=str(path),
                        reused=False,
                    )
                )
    designs = {(run.architecture, run.seed, run.tasks) for run in runs}
    if len(runs) != 108 or len({run.run_id for run in runs}) != 108 or len(designs) != 108:
        raise ValueError("task-geometry matrix must contain 108 unique model designs")
    return tuple(runs)


def _checkpoint_path(marker: Mapping[str, Any], marker_path: Path) -> Path:
    checkpoint = Path(str(marker["checkpoint"]))
    if not checkpoint.is_file():
        checkpoint = marker_path.parent / checkpoint.name
    return checkpoint


def _completed(run: GeometryRun, *, steps: int) -> bool:
    marker_path = Path(run.output_dir) / "completed.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = _checkpoint_path(marker, marker_path)
        expected_experiment_sha = _sha256(Path(run.experiment_config))
        return (
            marker.get("status") == "completed"
            and marker.get("global_step") == steps
            and marker.get("run_id") == run.canonical_run_id
            and marker.get("architecture") == run.architecture
            and marker.get("tasks") == list(run.tasks)
            and marker.get("seed") == run.seed
            and marker.get("experiment_config_sha256") == expected_experiment_sha
            and checkpoint.is_file()
            and marker.get("checkpoint_sha256") == _sha256(checkpoint)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def geometry_summary(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(path)
    design = validate_geometry_config(path)
    runs = build_geometry_matrix(path)
    steps = int(config["training"]["max_steps"])
    completed = [run.run_id for run in runs if _completed(run, steps=steps)]
    by_kind: dict[str, dict[str, int]] = {}
    for kind in ("specialist", "bundle"):
        selected = [run for run in runs if run.kind == kind]
        count = sum(run.run_id in completed for run in selected)
        by_kind[kind] = {
            "run_count": len(selected),
            "complete_count": count,
            "incomplete_count": len(selected) - count,
        }
    return {
        "protocol_version": config["protocol_version"],
        "config_sha256": config_sha256,
        "run_count": len(runs),
        "complete_count": len(completed),
        "incomplete_count": len(runs) - len(completed),
        "design": design,
        "by_kind": by_kind,
        "complete": completed,
        "runs": [asdict(run) for run in runs],
    }


def _training_command(run: GeometryRun, config: Mapping[str, Any]) -> list[str]:
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
        run.experiment_config,
        "--resume",
    ]


def run_geometry_matrix(
    path: Path = DEFAULT_CONFIG,
    *,
    worker_index: int = 0,
    worker_count: int = 1,
    dry_run: bool = False,
) -> None:
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("worker index must satisfy 0 <= index < worker count")
    config, _ = _read_config(path)
    runs = build_geometry_matrix(path)
    steps = int(config["training"]["max_steps"])
    for index, run in enumerate(runs):
        if index % worker_count != worker_index:
            continue
        if _completed(run, steps=steps):
            continue
        command = _training_command(run, config)
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
    action.add_argument("--verify-relations", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--max-n", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_relations:
        result = verify_relation_identities(args.config, max_n=args.max_n)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.plan or args.status:
        print(json.dumps(geometry_summary(args.config), indent=2, sort_keys=True))
    else:
        run_geometry_matrix(
            args.config,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BUNDLE_SPLIT_IDS",
    "DEFAULT_CONFIG",
    "GeometryRun",
    "MODEL_SEEDS",
    "RELATED_PAIR_COUNTS",
    "TaskRelation",
    "build_geometry_matrix",
    "geometry_summary",
    "main",
    "relation_definitions",
    "run_geometry_matrix",
    "transform_permutation",
    "validate_geometry_config",
    "verify_relation_identities",
]

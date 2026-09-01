"""Orchestrate and validate the three zero-overlap property replicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any, Sequence

from .math_ops import PROPERTY32_TASK_NAMES
from .property_experiments import (
    build_property_matrix,
    matrix_summary,
    run_matrix,
)


REPLICATE_CONFIGS = {
    "r0": Path("configs/property32_zero_overlap_pilot.toml"),
    "r1": Path("configs/property32_zero_overlap_r1.toml"),
    "r2": Path("configs/property32_zero_overlap_r2.toml"),
}

PROPERTY_FAMILIES = (
    frozenset(
        {
            "descents",
            "recoils",
            "peaks",
            "valleys",
            "double_ascents",
            "double_descents",
            "successions",
            "adjacencies",
        }
    ),
    frozenset(
        {
            "fixed_points",
            "anti_fixed_points",
            "exceedances",
            "deficiencies",
            "left_to_right_maxima",
            "left_to_right_minima",
            "right_to_left_maxima",
            "right_to_left_minima",
        }
    ),
    frozenset(
        {
            "cycle_count",
            "two_cycle_count",
            "three_cycle_count",
            "even_cycle_count",
            "odd_cycle_count",
            "longest_cycle",
            "shortest_cycle",
            "nontrivial_cycle_count",
        }
    ),
    frozenset(
        {
            "lis_length",
            "lds_length",
            "longest_increasing_run",
            "longest_decreasing_run",
            "global_descents",
            "components",
            "max_displacement",
            "displacement_one_count",
        }
    ),
)

NATURAL_DUAL_PAIRS = (
    ("descents", "recoils"),
    ("peaks", "valleys"),
    ("double_ascents", "double_descents"),
    ("successions", "adjacencies"),
    ("fixed_points", "anti_fixed_points"),
    ("exceedances", "deficiencies"),
    ("left_to_right_maxima", "left_to_right_minima"),
    ("right_to_left_maxima", "right_to_left_minima"),
    ("cycle_count", "nontrivial_cycle_count"),
    ("two_cycle_count", "three_cycle_count"),
    ("even_cycle_count", "odd_cycle_count"),
    ("longest_cycle", "shortest_cycle"),
    ("lis_length", "lds_length"),
    ("longest_increasing_run", "longest_decreasing_run"),
    ("global_descents", "components"),
    ("max_displacement", "displacement_one_count"),
)


def _read_config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def validate_replicate_design() -> dict[str, Any]:
    """Validate the frozen splits and return their design-level provenance."""

    registry = set(PROPERTY32_TASK_NAMES)
    memberships = {task: [] for task in PROPERTY32_TASK_NAMES}
    result: dict[str, Any] = {"replicates": {}}
    expected_seeds = {"r0": 17, "r1": 42, "r2": 101}
    for replicate_id, path in REPLICATE_CONFIGS.items():
        config = _read_config(path)
        runs = build_property_matrix(path)
        pool_a = tuple(config["pool_a"])
        pool_b = tuple(config["pool_b"])
        if set(pool_a) | set(pool_b) != registry or set(pool_a) & set(pool_b):
            raise ValueError(f"{replicate_id} pools do not partition all properties")
        if tuple(config["model_seeds"]) != (expected_seeds[replicate_id],):
            raise ValueError(f"{replicate_id} has the wrong frozen model seed")
        for task in PROPERTY32_TASK_NAMES:
            memberships[task].append("a" if task in pool_a else "b")
        for pool in (pool_a, pool_b):
            for block_start in (0, 4, 8, 12):
                block = set(pool[block_start : block_start + 4])
                if [len(block & family) for family in PROPERTY_FAMILIES] != [1] * 4:
                    raise ValueError(
                        f"{replicate_id} does not balance every four-task block"
                    )
        across = sum(
            (left in pool_a) != (right in pool_a)
            for left, right in NATURAL_DUAL_PAIRS
        )
        result["replicates"][replicate_id] = {
            "config": str(path),
            "model_seed": expected_seeds[replicate_id],
            "pool_split_seed": config.get("pool_split_seed"),
            "dual_pairs_across_pools": across,
            "dual_pairs_within_pools": len(NATURAL_DUAL_PAIRS) - across,
            "run_count": len(runs),
        }
    if any(set(sides) != {"a", "b"} for sides in memberships.values()):
        raise ValueError("every property must change pool in the replicate design")
    result["property_memberships"] = memberships
    result["run_count"] = 30
    return result


def combined_status() -> dict[str, Any]:
    design = validate_replicate_design()
    summaries = {
        replicate_id: matrix_summary(path)
        for replicate_id, path in REPLICATE_CONFIGS.items()
    }
    return {
        "run_count": sum(value["run_count"] for value in summaries.values()),
        "complete_count": sum(
            value["complete_count"] for value in summaries.values()
        ),
        "incomplete_count": sum(
            value["incomplete_count"] for value in summaries.values()
        ),
        "design": design,
        "replicates": summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate", choices=("all", *REPLICATE_CONFIGS), default="all"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--status", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_replicate_design()
    selected = (
        REPLICATE_CONFIGS
        if args.replicate == "all"
        else {args.replicate: REPLICATE_CONFIGS[args.replicate]}
    )
    if args.status:
        payload = combined_status() if args.replicate == "all" else {
            args.replicate: matrix_summary(selected[args.replicate])
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for path in selected.values():
        run_matrix(path, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NATURAL_DUAL_PAIRS",
    "PROPERTY_FAMILIES",
    "REPLICATE_CONFIGS",
    "combined_status",
    "main",
    "validate_replicate_design",
]


from __future__ import annotations

from pathlib import Path
import tomllib

from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES
from neurips_permutations.property_experiments import build_property_matrix
from neurips_permutations.property_replicates import (
    NATURAL_DUAL_PAIRS,
    PROPERTY_FAMILIES,
)


def test_fixed_seed_subset_replicates_are_balanced_and_disjoint() -> None:
    for replicate_id, split_seed in (("r3", 20261101), ("r4", 20261359)):
        path = Path("configs") / f"property32_zero_overlap_{replicate_id}.toml"
        config = tomllib.loads(path.read_text(encoding="utf-8"))
        pool_a = tuple(config["pool_a"])
        pool_b = tuple(config["pool_b"])
        assert config["model_seeds"] == [17]
        assert config["pool_split_seed"] == split_seed
        assert set(pool_a).isdisjoint(pool_b)
        assert set(pool_a) | set(pool_b) == set(PROPERTY32_TASK_NAMES)
        for pool in (pool_a, pool_b):
            for start in (0, 4, 8, 12):
                block = set(pool[start : start + 4])
                assert [len(block & family) for family in PROPERTY_FAMILIES] == [1] * 4
        assert sum(
            (left in pool_a) != (right in pool_a)
            for left, right in NATURAL_DUAL_PAIRS
        ) == 8
        runs = build_property_matrix(path)
        assert len(runs) == len({run.run_id for run in runs}) == 10

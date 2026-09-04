from __future__ import annotations

import csv
import json
from pathlib import Path

from neurips_permutations.property_replicate_results import BEHAVIOR_METRICS
from neurips_permutations.property_subset_replicate_results import (
    CONFIGS,
    build_behavior_rows,
    build_cka_rows,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_fixed_seed_builders_keep_split_replicates_separate(tmp_path: Path) -> None:
    directories = {replicate: tmp_path / replicate for replicate in CONFIGS}
    for offset, (replicate, directory) in enumerate(directories.items()):
        behavior = []
        for task_count in (1, 2, 4, 8, 16):
            for pool in ("A", "B"):
                row: dict[str, object] = {
                    "pool": pool,
                    "trained_task_count": task_count,
                    "task_status": "opposite_pool",
                }
                for metric in BEHAVIOR_METRICS:
                    row[metric] = offset + task_count / 100
                behavior.append(row)
        _write_csv(directory / "behavior" / "SUMMARY.csv", behavior)
        cka = {
            "primary_trend": {
                "task_counts": [1, 2, 4, 8, 16],
                "final_layer_linear_cka": [offset + value / 100 for value in (1, 2, 4, 8, 16)],
            }
        }
        path = directory / "cka" / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cka), encoding="utf-8")
    behavior_rows = build_behavior_rows(directories)
    cka_rows = build_cka_rows(directories)
    assert len(behavior_rows) == len(cka_rows) == 15
    assert {row["model_seed"] for row in behavior_rows} == {17}
    assert {row["replicate_id"] for row in cka_rows} == {"r0", "r3", "r4"}

"""Unit tests for the zero-overlap property CKA analysis."""

from __future__ import annotations

import torch

from neurips_permutations.cka import ActivationSet
from neurips_permutations.property_cka import (
    build_property_cka_rows,
    summarize_primary_trend,
)


def _activation(pool: str, task_count: int, seed: int = 17) -> ActivationSet:
    generator = torch.Generator().manual_seed(task_count + (0 if pool == "a" else 100))
    return ActivationSet(
        run_id=f"{pool}-{task_count}-{seed}",
        architecture="transformer",
        task_count=task_count,
        seed=seed,
        checkpoint_sha256="fixture",
        probe_sha256="probe",
        layers={"final_norm": torch.randn(12, 5, generator=generator)},
    )


def test_pair_builder_has_primary_overlap_control_and_random_rows() -> None:
    activations = [
        _activation(pool, task_count)
        for pool in ("a", "b")
        for task_count in (1, 2, 4, 8, 16)
    ]
    randoms = [_activation("random", 0, seed) for seed in (17, 42)]
    pools = {item.run_id: item.run_id[0] for item in activations}
    rows = build_property_cka_rows(
        activations, pools, randoms, device=torch.device("cpu")
    )
    assert len(rows) == 14
    assert sum(row["comparison"] == "disjoint_pools_equal_k" for row in rows) == 5
    assert (
        sum(row["comparison"] == "within_pool_k16_alignment" for row in rows)
        == 8
    )
    assert sum(row["comparison"] == "random_cross_seed" for row in rows) == 1
    assert all(0.0 <= row["linear_cka"] <= 1.000001 for row in rows)


def test_primary_summary_reports_monotonic_trend() -> None:
    rows = [
        {
            "comparison": "disjoint_pools_equal_k",
            "layer": "final_norm",
            "task_count_a": task_count,
            "linear_cka": value,
        }
        for task_count, value in zip(
            (1, 2, 4, 8, 16), (0.2, 0.3, 0.4, 0.6, 0.8)
        )
    ]
    summary = summarize_primary_trend(rows)
    assert summary["monotonic_non_decreasing"] is True
    assert summary["spearman_rho"] == 1.0
    assert abs(summary["delta_k16_minus_k1"] - 0.6) < 1e-12

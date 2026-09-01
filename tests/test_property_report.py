import pytest

from neurips_permutations.property_experiments import build_property_matrix
from neurips_permutations.property_report import summarize_rows, task_status


def test_task_status_separates_seen_and_two_unseen_groups():
    runs = build_property_matrix()
    a4 = next(run for run in runs if run.pool == "a" and run.task_count == 4)
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        Path("configs/property32_zero_overlap_pilot.toml").read_text()
    )
    assert task_status(
        a4, "descents", pool_a=config["pool_a"], pool_b=config["pool_b"]
    ) == "seen"
    assert task_status(
        a4, "peaks", pool_a=config["pool_a"], pool_b=config["pool_b"]
    ) == "same_pool_unseen"
    assert task_status(
        a4, "recoils", pool_a=config["pool_a"], pool_b=config["pool_b"]
    ) == "opposite_pool"


def test_summary_is_task_macro_not_token_weighted():
    rows = [
        {
            "pool": "A",
            "trained_task_count": 1,
            "task_status": "opposite_pool",
            "loss": 1.0,
            "token_accuracy": 0.2,
            "sequence_accuracy": 0.1,
            "supervised_tokens": 2,
        },
        {
            "pool": "A",
            "trained_task_count": 1,
            "task_status": "opposite_pool",
            "loss": 3.0,
            "token_accuracy": 0.8,
            "sequence_accuracy": 0.5,
            "supervised_tokens": 200,
        },
    ]
    summary = summarize_rows(rows)
    assert len(summary) == 1
    assert summary[0]["task_count"] == 2
    assert summary[0]["macro_loss"] == pytest.approx(2.0)
    assert summary[0]["macro_token_accuracy"] == pytest.approx(0.5)
    assert summary[0]["macro_sequence_accuracy"] == pytest.approx(0.3)

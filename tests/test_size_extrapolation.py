from __future__ import annotations

import math

from neurips_permutations.size_extrapolation import _status, build_summary, evaluation_tasks


def test_extrapolation_excludes_only_context_overflow_task() -> None:
    tasks = evaluation_tasks()
    assert len(tasks) == 19
    assert "to_reduced_word" not in tasks
    assert "compose" in tasks and "parity" in tasks
    assert _status("length", ("length",)) == "seen"
    assert _status("compose", ()) == "fixed_train_holdout"
    assert _status("length", ()) == "pool_unseen"


def test_extrapolation_summary_averages_tasks_within_seed_first() -> None:
    rows = []
    for seed_index, seed in enumerate((17, 42, 314159)):
        for task in ("length", "descents"):
            value = 0.1 + 0.1 * seed_index
            rows.append(
                {
                    "architecture": "transformer",
                    "trained_task_count": 2,
                    "task_status": "seen",
                    "seed": seed,
                    "task": task,
                    "loss": 1.0,
                    "token_accuracy": value,
                    "sequence_accuracy": value,
                    "in_domain_sequence_accuracy": value + 0.2,
                    "sequence_accuracy_delta_out_minus_in": -0.2,
                }
            )
    summary = build_summary(rows)
    assert len(summary) == 1
    assert summary[0]["task_count"] == 2
    assert math.isclose(summary[0]["sequence_accuracy_mean"], 0.2)
    assert math.isclose(summary[0]["sequence_accuracy_delta_out_minus_in_mean"], -0.2)

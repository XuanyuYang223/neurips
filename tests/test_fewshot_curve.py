from __future__ import annotations

import math

from neurips_permutations.fewshot_curve import build_endpoint_delta


TASKS = ("to_reduced_word", "compose", "parity", "to_lehmer")


def test_endpoint_delta_pairs_shots_tasks_and_seeds_before_averaging() -> None:
    rows = []
    for shots in (5, 20, 100):
        for initialization, task_count in (("pretrained", 4), ("random", 0)):
            for seed_index, seed in enumerate((17, 42, 314159)):
                for task_index, task in enumerate(TASKS):
                    endpoint_effect = 0.01 * seed_index + 0.001 * task_index
                    rows.append(
                        {
                            "shots": shots,
                            "initialization": initialization,
                            "architecture": "transformer",
                            "base_trained_task_count": task_count,
                            "seed": seed,
                            "task": task,
                            "loss": 2.0 - (0.5 if shots == 100 else 0.0),
                            "token_accuracy": endpoint_effect
                            + (0.2 if shots == 100 else 0.0),
                            "sequence_accuracy": endpoint_effect
                            + (0.1 if shots == 100 else 0.0),
                        }
                    )
    result = build_endpoint_delta(rows)
    assert len(result) == 2
    for row in result:
        assert math.isclose(row["loss_100_minus_5_mean"], -0.5)
        assert math.isclose(row["token_accuracy_100_minus_5_mean"], 0.2)
        assert math.isclose(row["sequence_accuracy_100_minus_5_mean"], 0.1)
        assert row["seed_count"] == 3
        assert row["task_count"] == 4

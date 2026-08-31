from __future__ import annotations

import math

from neurips_permutations.fewshot_results import (
    RAW_FIELDS,
    build_gains,
    build_summary,
)


TASKS = ("to_reduced_word", "compose", "parity", "to_lehmer")
SEEDS = (17, 42, 314159)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(("transformer", "mlp")):
        for seed_index, seed in enumerate(SEEDS):
            for task_index, task in enumerate(TASKS):
                random_loss = 8.0 + architecture_index + task_index / 10
                random_token = 0.1 + seed_index / 100
                random_sequence = 0.01 * task_index
                rows.append(
                    {
                        "initialization": "random",
                        "architecture": architecture,
                        "base_trained_task_count": 0,
                        "seed": seed,
                        "task": task,
                        "loss": random_loss,
                        "token_accuracy": random_token,
                        "sequence_accuracy": random_sequence,
                    }
                )
                for count in (1, 2, 4, 8, 16):
                    rows.append(
                        {
                            "initialization": "pretrained",
                            "architecture": architecture,
                            "base_trained_task_count": count,
                            "seed": seed,
                            "task": task,
                            "loss": random_loss - count / 10,
                            "token_accuracy": random_token + count / 100,
                            "sequence_accuracy": random_sequence + count / 1000,
                            "loss_improvement_from_zero_shot": count / 5,
                            "token_accuracy_improvement_from_zero_shot": count / 200,
                            "sequence_accuracy_improvement_from_zero_shot": count / 2000,
                        }
                    )
    return rows


def test_summary_keeps_random_and_pretrained_conditions_separate() -> None:
    summary = build_summary(_rows())
    assert len(summary) == 12
    assert sum(row["initialization"] == "pretrained" for row in summary) == 10
    assert sum(row["initialization"] == "random" for row in summary) == 2
    assert {row["seed_count"] for row in summary} == {3}
    assert {row["task_count"] for row in summary} == {4}


def test_paired_gains_are_macro_averaged_by_seed_then_reported() -> None:
    gains = build_gains(_rows())
    assert len(gains) == 10
    transformer_k16 = next(
        row
        for row in gains
        if row["architecture"] == "transformer"
        and row["base_trained_task_count"] == 16
    )
    assert math.isclose(transformer_k16["loss_improvement_over_random_mean"], 1.6)
    assert math.isclose(
        transformer_k16["token_accuracy_improvement_over_random_mean"], 0.16
    )
    assert math.isclose(
        transformer_k16["sequence_accuracy_improvement_from_zero_shot_mean"],
        0.008,
    )
    assert transformer_k16["loss_improvement_over_random_sample_sd"] == 0


def test_public_raw_schema_does_not_restore_redundant_protocol_column() -> None:
    assert RAW_FIELDS[0] == "run_id"
    assert "protocol_version" not in RAW_FIELDS

import pytest

from neurips_permutations.property_replicate_results import (
    BEHAVIOR_METRICS,
    summarize_behavior_replicates,
    summarize_cka_replicates,
)


def test_behavior_summary_uses_three_replicates_as_units() -> None:
    rows = []
    for index, replicate_id in enumerate(("r0", "r1", "r2"), start=1):
        for task_count in (1, 2, 4, 8, 16):
            row = {
                "replicate_id": replicate_id,
                "trained_task_count": task_count,
            }
            row.update({metric: index / 10 for metric in BEHAVIOR_METRICS})
            rows.append(row)
    summary = summarize_behavior_replicates(rows)
    assert len(summary) == 5
    assert all(row["replicate_count"] == 3 for row in summary)
    assert summary[0]["macro_sequence_accuracy_mean"] == pytest.approx(0.2)
    assert summary[0]["macro_sequence_accuracy_sample_sd"] == pytest.approx(0.1)


def test_cka_summary_reports_mean_sd_and_replicate_peaks() -> None:
    rows = []
    values = {
        "r0": (0.1, 0.2, 0.3, 0.8, 0.4),
        "r1": (0.2, 0.3, 0.4, 0.5, 0.6),
        "r2": (0.3, 0.4, 0.5, 0.7, 0.6),
    }
    for replicate_id, sequence in values.items():
        for task_count, value in zip((1, 2, 4, 8, 16), sequence):
            rows.append(
                {
                    "replicate_id": replicate_id,
                    "trained_task_count": task_count,
                    "final_layer_linear_cka": value,
                }
            )
    summary, trend = summarize_cka_replicates(rows)
    assert len(summary) == 5
    assert summary[0]["final_layer_linear_cka_mean"] == pytest.approx(0.2)
    assert summary[0]["final_layer_linear_cka_sample_sd"] == pytest.approx(0.1)
    assert trend["replicate_peak_k"] == {"r0": 8, "r1": 16, "r2": 8}
    assert trend["k8_peak_replicate_count"] == 2

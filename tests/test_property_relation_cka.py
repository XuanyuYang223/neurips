import pytest

from neurips_permutations.property_relation_cka import summarize_cell_curves
from neurips_permutations.property_relation_experiments import (
    MODEL_SEEDS,
    SPLIT_IDS,
    TASK_COUNTS,
)


def _rows(curves: dict[tuple[str, int], tuple[float, ...]]) -> list[dict[str, object]]:
    return [
        {
            "split_id": split_id,
            "model_seed": seed,
            "trained_task_count": task_count,
            "final_layer_linear_cka": value,
        }
        for (split_id, seed), values in curves.items()
        for task_count, value in zip(TASK_COUNTS, values)
    ]


def test_relation_cka_summary_uses_nine_cells_as_units() -> None:
    curves = {
        (split_id, seed): (0.1, 0.2, 0.3, 0.4 + 0.01 * index)
        for index, (split_id, seed) in enumerate(
            (pair for pair in ((s, seed) for s in SPLIT_IDS for seed in MODEL_SEEDS))
        )
    }
    summary, trend = summarize_cell_curves(_rows(curves))
    assert len(summary) == 4
    assert all(row["cell_count"] == 9 for row in summary)
    assert summary[0]["final_layer_linear_cka_mean"] == pytest.approx(0.1)
    assert trend["positive_delta_cells"] == 9
    assert trend["two_sided_exact_sign_test_p"] == pytest.approx(2 / 512)
    assert trend["monotonic_cell_count"] == 9
    assert set(trend["split_mean_deltas"]) == set(SPLIT_IDS)
    assert set(trend["seed_mean_deltas"]) == {"17", "42", "101"}


def test_relation_cka_summary_rejects_missing_cell() -> None:
    curves = {
        (split_id, seed): (0.1, 0.2, 0.3, 0.4)
        for split_id in SPLIT_IDS
        for seed in MODEL_SEEDS
    }
    rows = _rows(curves)
    with pytest.raises(ValueError, match="incomplete"):
        summarize_cell_curves(rows[:-1])

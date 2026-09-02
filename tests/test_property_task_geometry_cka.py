from itertools import combinations

import pytest

from neurips_permutations.cka import ProbeExample
from neurips_permutations.passage import TOKEN_TO_ID
from neurips_permutations.property_task_geometry_cka import (
    _symmetry_summary,
    summarize_bundle_rows,
    summarize_specialist_rows,
    task_label_permutation_test,
    transform_probe_examples,
)
from neurips_permutations.property_task_geometry_experiments import (
    BUNDLE_SPLIT_IDS,
    MODEL_SEEDS,
    RELATED_PAIR_COUNTS,
    relation_definitions,
)


def test_probe_transformations_rebuild_canonical_one_line_prefix() -> None:
    tokens = (
        "<BOS>",
        "<SIZE>",
        "04",
        "<ONE_START>",
        "02",
        ",",
        "01",
        ",",
        "04",
        ",",
        "03",
        "<ONE_END>",
    )
    example = ProbeExample(
        record_id=9,
        n=4,
        token_ids=tuple(TOKEN_TO_ID[token] for token in tokens),
    )
    inverse = transform_probe_examples((example,), "inverse")[0]
    complement = transform_probe_examples((example,), "complement")[0]
    inverse_tokens = tuple(
        next(token for token, index in TOKEN_TO_ID.items() if index == value)
        for value in inverse.token_ids
    )
    complement_tokens = tuple(
        next(token for token, index in TOKEN_TO_ID.items() if index == value)
        for value in complement.token_ids
    )
    assert inverse_tokens[4:-1:2] == ("02", "01", "04", "03")
    assert complement_tokens[4:-1:2] == ("03", "04", "01", "02")
    assert inverse.record_id == complement.record_id == 9


def test_bundle_summary_uses_twelve_paired_cells() -> None:
    rows = [
        {
            "split_id": split_id,
            "model_seed": seed,
            "related_pair_count": count,
            "layer": "final_norm",
            "linear_cka": 0.1 + 0.1 * RELATED_PAIR_COUNTS.index(count),
        }
        for split_id in BUNDLE_SPLIT_IDS
        for seed in MODEL_SEEDS
        for count in RELATED_PAIR_COUNTS
    ]
    summary, trend = summarize_bundle_rows(rows)
    assert [row["cell_count"] for row in summary] == [12, 12, 12, 12]
    assert trend["positive_delta_cells"] == 12
    assert trend["monotonic_cell_count"] == 12
    assert trend["delta_r4_minus_r0_mean"] == pytest.approx(0.3)
    assert trend["two_sided_exact_sign_test_p"] == pytest.approx(2 / 4096)


def test_specialist_summary_aggregates_seed_values_before_task_pairs() -> None:
    rows = []
    for comparison, pairs, value in (
        ("same_task", (("a", "a"), ("b", "b")), 0.9),
        ("direct_relation", (("a", "b"),), 0.6),
        ("no_direct_relation", (("a", "c"), ("b", "c")), 0.2),
    ):
        for task_a, task_b in pairs:
            for seed_a, seed_b in combinations(MODEL_SEEDS, 2):
                rows.append(
                    {
                        "comparison": comparison,
                        "task_a": task_a,
                        "task_b": task_b,
                        "same_seed": False,
                        "layer": "final_norm",
                        "linear_cka": value,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                    }
                )
    pair_rows, group_rows = summarize_specialist_rows(rows)
    assert len(pair_rows) == 5
    groups = {row["comparison"]: row for row in group_rows}
    assert groups["same_task"]["task_pair_count"] == 2
    assert groups["direct_relation"]["final_layer_cka_mean"] == pytest.approx(0.6)
    assert groups["no_direct_relation"]["task_pair_count"] == 2


def test_task_label_permutation_test_is_deterministic_and_complete() -> None:
    relations = relation_definitions()
    tasks = tuple(task for relation in relations for task in (relation.left, relation.right))
    related = {frozenset((relation.left, relation.right)) for relation in relations}
    rows = [
        {
            "comparison": "direct_relation"
            if frozenset((left, right)) in related
            else "no_direct_relation",
            "task_a": left,
            "task_b": right,
            "final_layer_cka_mean": 0.8
            if frozenset((left, right)) in related
            else 0.2,
        }
        for left, right in combinations(tasks, 2)
    ]
    first = task_label_permutation_test(
        rows, relations, permutations_count=100, seed=123
    )
    second = task_label_permutation_test(
        rows, relations, permutations_count=100, seed=123
    )
    assert first == second
    assert first["observed"] == pytest.approx(0.6)
    assert 0 < first["one_sided_p"] <= 1


def test_symmetry_summary_reports_exact_paired_sign_tests() -> None:
    rows = [
        {
            "pair_id": pair_id,
            "model_seed": seed,
            "condition": condition,
            "layer": "final_norm",
            "linear_cka": value,
        }
        for pair_id in ("pair-a", "pair-b")
        for seed in MODEL_SEEDS
        for condition, value in (
            ("identity", 0.2),
            ("correct", 0.8),
            ("wrong", 0.1),
        )
    ]
    _, trend = _symmetry_summary(rows)
    assert trend["pair_seed_units"] == 6
    assert trend["positive_correct_minus_identity_units"] == 6
    assert trend["positive_correct_minus_wrong_units"] == 6
    assert trend[
        "two_sided_exact_sign_test_p_correct_minus_identity"
    ] == pytest.approx(2 / 64)
    assert trend[
        "two_sided_exact_sign_test_p_correct_minus_wrong"
    ] == pytest.approx(2 / 64)
    assert trend["relation_units"] == 2
    assert trend["positive_relation_mean_correct_minus_identity"] == 2
    assert trend["positive_relation_mean_correct_minus_wrong"] == 2
    assert trend[
        "two_sided_relation_level_sign_test_p_correct_minus_identity"
    ] == pytest.approx(2 / 4)
    assert trend[
        "two_sided_relation_level_sign_test_p_correct_minus_wrong"
    ] == pytest.approx(2 / 4)
    assert trend["relation_mean_deltas"]["pair-a"] == pytest.approx(
        {"correct_minus_identity": 0.6, "correct_minus_wrong": 0.7}
    )

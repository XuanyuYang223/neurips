"""Tests for the scalar-only, zero-overlap 32-property protocol."""

from __future__ import annotations

from itertools import permutations

import neurips_permutations as package
from neurips_permutations import math_ops as ops
from neurips_permutations.passage import TASK_SPECS, passage_tokens


POOL_A = (
    "descents",
    "fixed_points",
    "cycle_count",
    "lis_length",
    "peaks",
    "exceedances",
    "three_cycle_count",
    "longest_increasing_run",
    "double_ascents",
    "left_to_right_maxima",
    "odd_cycle_count",
    "global_descents",
    "successions",
    "right_to_left_maxima",
    "longest_cycle",
    "max_displacement",
)

POOL_B = (
    "recoils",
    "anti_fixed_points",
    "two_cycle_count",
    "lds_length",
    "valleys",
    "deficiencies",
    "even_cycle_count",
    "longest_decreasing_run",
    "double_descents",
    "left_to_right_minima",
    "shortest_cycle",
    "components",
    "adjacencies",
    "right_to_left_minima",
    "nontrivial_cycle_count",
    "displacement_one_count",
)


def test_registry_is_complete_exported_and_zero_overlap() -> None:
    assert package.PROPERTY32_TASK_NAMES is ops.PROPERTY32_TASK_NAMES
    assert tuple(ops.PROPERTY_FUNCTIONS) == ops.PROPERTY32_TASK_NAMES
    assert len(ops.PROPERTY32_TASK_NAMES) == 32
    assert len(set(ops.PROPERTY32_TASK_NAMES)) == 32
    assert len(POOL_A) == len(POOL_B) == 16
    assert set(POOL_A).isdisjoint(POOL_B)
    assert set(POOL_A) | set(POOL_B) == set(ops.PROPERTY32_TASK_NAMES)
    for count in (1, 2, 4, 8, 16):
        assert set(POOL_A[:count]).isdisjoint(POOL_B[:count])


def test_reference_fixture_has_expected_values() -> None:
    fixture = (3, 1, 4, 2)
    expected = {
        "descents": 2,
        "recoils": 1,
        "peaks": 1,
        "valleys": 1,
        "double_ascents": 0,
        "double_descents": 0,
        "successions": 0,
        "adjacencies": 0,
        "fixed_points": 0,
        "anti_fixed_points": 0,
        "exceedances": 2,
        "deficiencies": 2,
        "left_to_right_maxima": 2,
        "left_to_right_minima": 2,
        "right_to_left_maxima": 2,
        "right_to_left_minima": 2,
        "cycle_count": 1,
        "two_cycle_count": 0,
        "three_cycle_count": 0,
        "even_cycle_count": 1,
        "odd_cycle_count": 0,
        "longest_cycle": 4,
        "shortest_cycle": 4,
        "nontrivial_cycle_count": 1,
        "lis_length": 2,
        "lds_length": 2,
        "longest_increasing_run": 2,
        "longest_decreasing_run": 2,
        "global_descents": 0,
        "components": 1,
        "max_displacement": 2,
        "displacement_one_count": 2,
    }
    assert {
        task: function(fixture)
        for task, function in ops.PROPERTY_FUNCTIONS.items()
    } == expected


def test_identity_and_reverse_extremes() -> None:
    identity = (1, 2, 3, 4)
    reverse = (4, 3, 2, 1)
    assert ops.double_ascent_count(identity) == 2
    assert ops.succession_count(identity) == 3
    assert ops.fixed_point_count(identity) == 4
    assert ops.component_count(identity) == 4
    assert ops.longest_increasing_run_length(identity) == 4
    assert ops.global_descent_count(reverse) == 3
    assert ops.anti_fixed_point_count(reverse) == 4
    assert ops.two_cycle_count(reverse) == 2
    assert ops.longest_decreasing_run_length(reverse) == 4
    assert ops.component_count(reverse) == 1


def test_all_small_permutations_are_bounded_and_satisfy_identities() -> None:
    for size in range(7):
        for permutation in permutations(range(1, size + 1)):
            values = {
                task: function(permutation)
                for task, function in ops.PROPERTY_FUNCTIONS.items()
            }
            assert all(type(value) is int for value in values.values())
            assert all(0 <= value <= size for value in values.values())
            assert (
                values["fixed_points"]
                + values["exceedances"]
                + values["deficiencies"]
                == size
            )
            assert (
                values["even_cycle_count"] + values["odd_cycle_count"]
                == values["cycle_count"]
            )
            assert (
                values["nontrivial_cycle_count"] + values["fixed_points"]
                == values["cycle_count"]
            )
            assert values["recoils"] == ops.descent_count(
                ops.inverse(permutation)
            )


def test_every_property_uses_one_scalar_answer_token() -> None:
    primary = (3, 1, 4, 2)
    for task, function in ops.PROPERTY_FUNCTIONS.items():
        assert TASK_SPECS[task].answer_kind == "scalar"
        tokens = passage_tokens(task, primary, function(primary))
        equals = tokens.index("=")
        assert tokens[equals + 2] == "<EOS>"
        assert tokens[equals + 1].isdigit() and len(tokens[equals + 1]) == 2
        assert "<NUM_START>" not in tokens[equals + 1 :]

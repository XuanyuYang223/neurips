"""Fixture and exhaustive small-n tests for permutation mathematics."""

from __future__ import annotations

import unittest
from itertools import combinations, permutations

import neurips_permutations as package
from neurips_permutations import math_ops as ops


def _identity(size: int) -> tuple[int, ...]:
    return tuple(range(1, size + 1))


def _standardize(values: tuple[int, ...]) -> tuple[int, ...]:
    rank = {value: index for index, value in enumerate(sorted(values), start=1)}
    return tuple(rank[value] for value in values)


def _naive_contains(
    permutation: tuple[int, ...], pattern: tuple[int, ...]
) -> bool:
    return any(
        _standardize(tuple(permutation[index] for index in indices)) == pattern
        for indices in combinations(range(len(permutation)), len(pattern))
    )


class ValidationTests(unittest.TestCase):
    def test_valid_permutations_are_normalized_to_tuples(self) -> None:
        self.assertEqual(ops.validate_permutation([3, 1, 2]), (3, 1, 2))
        self.assertEqual(ops.validate_permutation(iter([2, 1])), (2, 1))
        self.assertEqual(ops.validate_permutation(()), ())

    def test_invalid_entry_sets_are_rejected(self) -> None:
        for malformed in ([1, 1], [0, 1], [2], [1, 3], [-1, 1]):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    ops.validate_permutation(malformed)

    def test_invalid_python_values_are_rejected(self) -> None:
        for malformed in ("123", 123, [True], [1.0], [1, "2"]):
            with self.subTest(malformed=malformed):
                with self.assertRaises(TypeError):
                    ops.validate_permutation(malformed)  # type: ignore[arg-type]


class TaskGridTests(unittest.TestCase):
    def test_task_grids_are_exported_from_the_package_root(self) -> None:
        self.assertIs(package.TASK_NAMES, ops.V2_TASK_NAMES)
        self.assertIs(package.V2_TASK_NAMES, ops.V2_TASK_NAMES)
        self.assertIs(package.V3_TASK_NAMES, ops.V3_TASK_NAMES)

    def test_v2_grid_remains_the_legacy_default(self) -> None:
        self.assertIs(ops.TASK_NAMES, ops.V2_TASK_NAMES)
        self.assertEqual(len(ops.V2_TASK_NAMES), 20)
        self.assertEqual(
            ops.V2_TASK_NAMES[15:18], ("power", "conjugate", "commutator")
        )

    def test_v3_grid_replaces_slow_algebra_tasks_in_category_order(self) -> None:
        self.assertEqual(
            ops.V3_TASK_NAMES,
            (
                "to_cycle",
                "to_lehmer",
                "to_inversion_vector",
                "to_reduced_word",
                "length",
                "descents",
                "fixed_points",
                "parity",
                "cycle_type",
                "rsk_shape",
                "lis_length",
                "lds_length",
                "pattern_avoidance",
                "peaks",
                "exceedances",
                "recoils",
                "inverse",
                "compose",
                "right_multiply_simple",
                "bruhat_leq",
            ),
        )
        self.assertEqual(len(ops.V3_TASK_NAMES), 20)
        self.assertEqual(
            set(ops.V2_TASK_NAMES) - set(ops.V3_TASK_NAMES),
            {"power", "conjugate", "commutator"},
        )
        self.assertEqual(
            set(ops.V3_TASK_NAMES) - set(ops.V2_TASK_NAMES),
            {"peaks", "exceedances", "recoils"},
        )


class RepresentationTests(unittest.TestCase):
    fixture = (3, 1, 4, 2)

    def test_passage_fixture(self) -> None:
        self.assertEqual(ops.canonical_cycles(self.fixture), ((1, 3, 4, 2),))
        self.assertEqual(ops.lehmer_code(self.fixture), (2, 0, 1, 0))
        self.assertEqual(ops.inversion_vector(self.fixture), (1, 2, 0, 0))
        self.assertEqual(ops.reduced_coxeter_word(self.fixture), (2, 3, 1))

    def test_cycles_include_singletons_and_are_canonical(self) -> None:
        self.assertEqual(
            ops.canonical_cycles((3, 4, 1, 2)), ((1, 3), (2, 4))
        )
        self.assertEqual(ops.canonical_cycles((2, 1, 3)), ((1, 2), (3,)))

    def test_encodings_are_bijective_and_have_correct_length(self) -> None:
        for size in range(6):
            seen_lehmer: set[tuple[int, ...]] = set()
            seen_inversion_vectors: set[tuple[int, ...]] = set()
            for permutation in permutations(range(1, size + 1)):
                with self.subTest(size=size, permutation=permutation):
                    inversion_count = ops.inversion_count(permutation)
                    lehmer = ops.lehmer_code(permutation)
                    vector = ops.inversion_vector(permutation)

                    self.assertEqual(sum(lehmer), inversion_count)
                    self.assertEqual(sum(vector), inversion_count)
                    self.assertTrue(
                        all(0 <= value <= size - index - 1 for index, value in enumerate(lehmer))
                    )
                    self.assertTrue(
                        all(0 <= value <= size - index - 1 for index, value in enumerate(vector))
                    )
                    seen_lehmer.add(lehmer)
                    seen_inversion_vectors.add(vector)

                    rebuilt = [0] * size
                    for cycle in ops.canonical_cycles(permutation):
                        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
                            rebuilt[source - 1] = target
                    self.assertEqual(tuple(rebuilt), permutation)

            expected_count = len(list(permutations(range(1, size + 1))))
            self.assertEqual(len(seen_lehmer), expected_count)
            self.assertEqual(len(seen_inversion_vectors), expected_count)

    def test_reduced_words_reconstruct_every_small_permutation(self) -> None:
        for size in range(6):
            identity = _identity(size)
            for permutation in permutations(range(1, size + 1)):
                with self.subTest(size=size, permutation=permutation):
                    word = ops.reduced_coxeter_word(permutation)
                    rebuilt = identity
                    for index in word:
                        self.assertTrue(1 <= index < size)
                        rebuilt = ops.right_multiply_simple(rebuilt, index)
                    self.assertEqual(rebuilt, permutation)
                    self.assertEqual(len(word), ops.inversion_count(permutation))


class StatisticTests(unittest.TestCase):
    def test_named_fixture(self) -> None:
        permutation = (3, 1, 4, 2)
        self.assertEqual(ops.inversion_count(permutation), 3)
        self.assertEqual(ops.descent_count(permutation), 2)
        self.assertEqual(ops.fixed_point_count(permutation), 0)
        self.assertEqual(ops.parity(permutation), 1)
        self.assertEqual(ops.cycle_type(permutation), (4,))
        self.assertEqual(ops.rsk_shape(permutation), (2, 2))
        self.assertEqual(ops.lis_length(permutation), 2)
        self.assertEqual(ops.lds_length(permutation), 2)
        self.assertEqual(ops.peak_count(permutation), 1)
        self.assertEqual(ops.exceedance_count(permutation), 2)
        self.assertEqual(ops.recoil_count(permutation), 1)

    def test_new_statistics_cover_empty_short_and_extremal_examples(self) -> None:
        for permutation in ((), (1,)):
            with self.subTest(permutation=permutation):
                self.assertEqual(ops.peak_count(permutation), 0)
                self.assertEqual(ops.exceedance_count(permutation), 0)
                self.assertEqual(ops.recoil_count(permutation), 0)

        self.assertEqual(ops.peak_count((2, 1)), 0)
        self.assertEqual(ops.exceedance_count((2, 1)), 1)
        self.assertEqual(ops.recoil_count((2, 1)), 1)

        self.assertEqual(ops.peak_count((1, 3, 2, 5, 4)), 2)
        self.assertEqual(ops.exceedance_count((2, 3, 4, 5, 1)), 4)
        self.assertEqual(ops.recoil_count((5, 4, 3, 2, 1)), 4)

    def test_new_statistics_match_the_passage_definitions_exhaustively(self) -> None:
        for size in range(7):
            for permutation in permutations(range(1, size + 1)):
                with self.subTest(size=size, permutation=permutation):
                    expected_peaks = sum(
                        permutation[index - 1]
                        < permutation[index]
                        > permutation[index + 1]
                        for index in range(1, size - 1)
                    )
                    expected_exceedances = sum(
                        value > index
                        for index, value in enumerate(permutation, start=1)
                    )
                    expected_recoils = ops.descent_count(ops.inverse(permutation))

                    self.assertEqual(ops.peak_count(permutation), expected_peaks)
                    self.assertEqual(
                        ops.exceedance_count(permutation), expected_exceedances
                    )
                    self.assertEqual(ops.recoil_count(permutation), expected_recoils)

    def test_new_statistics_reuse_strict_permutation_validation(self) -> None:
        for statistic in (
            ops.peak_count,
            ops.exceedance_count,
            ops.recoil_count,
        ):
            with self.subTest(statistic=statistic.__name__, error="value"):
                with self.assertRaises(ValueError):
                    statistic((1, 1))
            with self.subTest(statistic=statistic.__name__, error="type"):
                with self.assertRaises(TypeError):
                    statistic((True,))

    def test_cycle_type_keeps_fixed_points(self) -> None:
        self.assertEqual(ops.cycle_type((2, 1, 3, 5, 4)), (2, 2, 1))

    def test_rsk_schensted_correspondence_exhaustively(self) -> None:
        for size in range(6):
            for permutation in permutations(range(1, size + 1)):
                with self.subTest(size=size, permutation=permutation):
                    shape = ops.rsk_shape(permutation)
                    self.assertEqual(sum(shape), size)
                    self.assertTrue(
                        all(left >= right for left, right in zip(shape, shape[1:]))
                    )
                    expected_lis = shape[0] if shape else 0
                    self.assertEqual(ops.lis_length(permutation), expected_lis)
                    self.assertEqual(ops.lds_length(permutation), len(shape))
                    self.assertEqual(
                        ops.parity(permutation), ops.inversion_count(permutation) % 2
                    )
                    self.assertEqual(sum(ops.cycle_type(permutation)), size)


class PatternTests(unittest.TestCase):
    def test_classical_pattern_fixtures(self) -> None:
        permutation = (3, 1, 4, 2)
        self.assertTrue(ops.pattern_contains(permutation, (1, 3, 2)))
        self.assertFalse(ops.avoids_pattern(permutation, (1, 3, 2)))
        self.assertFalse(ops.pattern_contains(permutation, (3, 2, 1)))
        self.assertTrue(ops.avoids_pattern(permutation, (3, 2, 1)))
        self.assertTrue(ops.pattern_contains(permutation, ()))
        self.assertFalse(ops.avoids_pattern(permutation, ()))
        self.assertFalse(ops.pattern_contains((1, 2), (1, 2, 3)))

    def test_containment_matches_independent_enumeration(self) -> None:
        for size in range(1, 6):
            for permutation in permutations(range(1, size + 1)):
                for pattern_size in range(1, min(size, 3) + 1):
                    for pattern in permutations(range(1, pattern_size + 1)):
                        with self.subTest(
                            permutation=permutation, pattern=pattern
                        ):
                            expected = _naive_contains(permutation, pattern)
                            self.assertEqual(
                                ops.pattern_contains(permutation, pattern), expected
                            )
                            self.assertEqual(
                                ops.avoids_pattern(permutation, pattern), not expected
                            )


class AlgebraTests(unittest.TestCase):
    def test_composition_convention_and_noncommutativity(self) -> None:
        first = (2, 1, 3)
        second = (1, 3, 2)
        self.assertEqual(ops.compose(first, second), (2, 3, 1))
        self.assertEqual(ops.compose(second, first), (3, 1, 2))

    def test_inverse_and_associativity_exhaustively_on_s3(self) -> None:
        elements = list(permutations(range(1, 4)))
        identity = _identity(3)
        for permutation in elements:
            with self.subTest(permutation=permutation):
                inverse = ops.inverse(permutation)
                self.assertEqual(ops.compose(permutation, inverse), identity)
                self.assertEqual(ops.compose(inverse, permutation), identity)
                self.assertEqual(ops.inverse(inverse), permutation)
        for first in elements:
            for second in elements:
                for third in elements:
                    self.assertEqual(
                        ops.compose(ops.compose(first, second), third),
                        ops.compose(first, ops.compose(second, third)),
                    )

    def test_power(self) -> None:
        permutation = (2, 3, 1)
        self.assertEqual(ops.power(permutation, 0), (1, 2, 3))
        self.assertEqual(ops.power(permutation, 1), permutation)
        self.assertEqual(ops.power(permutation, 2), (3, 1, 2))
        self.assertEqual(ops.power(permutation, 3), (1, 2, 3))
        self.assertEqual(ops.power(permutation, 10**6), permutation)
        with self.assertRaises(ValueError):
            ops.power(permutation, -1)
        with self.assertRaises(TypeError):
            ops.power(permutation, True)

    def test_conjugation_and_commutator_conventions(self) -> None:
        first = (2, 1, 3)
        second = (1, 3, 2)
        self.assertEqual(ops.conjugate(second, first), (3, 2, 1))
        self.assertEqual(ops.commutator(first, second), (3, 1, 2))
        self.assertEqual(
            ops.cycle_type(ops.conjugate(second, first)), ops.cycle_type(first)
        )

    def test_right_simple_multiplication(self) -> None:
        permutation = (3, 1, 4, 2)
        self.assertEqual(ops.right_multiply_simple(permutation, 2), (3, 4, 1, 2))
        simple = (1, 3, 2, 4)
        self.assertEqual(
            ops.right_multiply_simple(permutation, 2),
            ops.compose(permutation, simple),
        )
        for bad_index in (0, 4):
            with self.assertRaises(ValueError):
                ops.right_multiply_simple(permutation, bad_index)
        with self.assertRaises(TypeError):
            ops.right_multiply_simple(permutation, True)

    def test_size_mismatch_is_rejected(self) -> None:
        for operation in (ops.compose, ops.conjugate, ops.commutator, ops.bruhat_leq):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(ValueError):
                    operation((1,), (1, 2))


class BruhatTests(unittest.TestCase):
    def test_known_s3_relations_include_strong_not_just_weak_order(self) -> None:
        identity = (1, 2, 3)
        s1 = (2, 1, 3)
        s2 = (1, 3, 2)
        s1s2 = (2, 3, 1)
        s2s1 = (3, 1, 2)
        longest = (3, 2, 1)

        for element in (identity, s1, s2, s1s2, s2s1, longest):
            self.assertTrue(ops.bruhat_leq(identity, element))
            self.assertTrue(ops.bruhat_leq(element, longest))
            self.assertTrue(ops.bruhat_leq(element, element))

        self.assertTrue(ops.bruhat_leq(s1, s2s1))
        self.assertTrue(ops.bruhat_leq(s2, s1s2))
        self.assertFalse(ops.bruhat_leq(s1, s2))
        self.assertFalse(ops.bruhat_leq(s2, s1))
        self.assertFalse(ops.bruhat_leq(s1s2, s2s1))
        self.assertFalse(ops.bruhat_leq(s2s1, s1s2))

    def test_order_axioms_and_length_monotonicity_on_s4(self) -> None:
        elements = list(permutations(range(1, 5)))
        identity = _identity(4)
        longest = tuple(reversed(identity))

        for first in elements:
            self.assertTrue(ops.bruhat_leq(first, first))
            self.assertTrue(ops.bruhat_leq(identity, first))
            self.assertTrue(ops.bruhat_leq(first, longest))
            for second in elements:
                first_leq_second = ops.bruhat_leq(first, second)
                if first_leq_second:
                    self.assertLessEqual(
                        ops.inversion_count(first), ops.inversion_count(second)
                    )
                if first_leq_second and ops.bruhat_leq(second, first):
                    self.assertEqual(first, second)

        for first in elements:
            for second in elements:
                if not ops.bruhat_leq(first, second):
                    continue
                for third in elements:
                    if ops.bruhat_leq(second, third):
                        self.assertTrue(ops.bruhat_leq(first, third))

    def test_right_ascent_gives_a_bruhat_cover(self) -> None:
        for permutation in permutations(range(1, 6)):
            for index in range(1, 5):
                if permutation[index - 1] < permutation[index]:
                    greater = ops.right_multiply_simple(permutation, index)
                    self.assertTrue(ops.bruhat_leq(permutation, greater))
                    self.assertEqual(
                        ops.inversion_count(greater),
                        ops.inversion_count(permutation) + 1,
                    )


if __name__ == "__main__":
    unittest.main()

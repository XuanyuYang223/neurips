"""Dependency-free permutation primitives used by the dataset generator.

Permutations use one-line notation with one-based values: ``p[i - 1]`` is
``p(i)`` and an element of ``S_n`` contains each integer in ``1, ..., n``
exactly once.  Results are immutable tuples so callers cannot accidentally
mutate a validated result.

Composition follows the function convention ``compose(a, b) = a ∘ b``.
Consequently, right multiplication by the simple transposition ``s_i`` swaps
positions ``i`` and ``i + 1`` in one-line notation.  Coxeter-word indices are
one-based and are to be applied to the identity by successive right
multiplications.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from itertools import combinations


Permutation = tuple[int, ...]
Cycle = tuple[int, ...]

# Kept here (rather than in the serializer) so generation, verification, and
# the package root share authoritative, ordered task grids.  ``TASK_NAMES``
# remains the v2 grid for backward compatibility with materialized v2 data and
# audit metadata; new v3 callers must opt in explicitly via ``V3_TASK_NAMES``.
V2_TASK_NAMES = (
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
    "inverse",
    "compose",
    "power",
    "conjugate",
    "commutator",
    "right_multiply_simple",
    "bruhat_leq",
)

V3_TASK_NAMES = (
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
)

TASK_NAMES = V2_TASK_NAMES

__all__ = [
    "Permutation",
    "TASK_NAMES",
    "V2_TASK_NAMES",
    "V3_TASK_NAMES",
    "avoids_pattern",
    "bruhat_leq",
    "canonical_cycles",
    "commutator",
    "compose",
    "conjugate",
    "cycle_type",
    "descent_count",
    "exceedance_count",
    "fixed_point_count",
    "inverse",
    "inversion_count",
    "inversion_vector",
    "lds_length",
    "lehmer_code",
    "lis_length",
    "parity",
    "peak_count",
    "contains_pattern",
    "pattern_contains",
    "power",
    "reduced_coxeter_word",
    "recoil_count",
    "right_multiply_simple",
    "rsk_shape",
    "validate_permutation",
]


def validate_permutation(permutation: Iterable[int]) -> Permutation:
    """Validate and return a permutation as a tuple.

    ``()`` is accepted as the unique permutation in ``S_0``.  Booleans and
    non-built-in integers are rejected rather than silently treating them as
    labels.  A ``TypeError`` denotes a malformed Python value and a
    ``ValueError`` denotes integer entries that are not exactly ``1, ..., n``.
    """

    if isinstance(permutation, (str, bytes)):
        raise TypeError("a permutation must be an iterable of integers")
    try:
        result = tuple(permutation)
    except TypeError as exc:
        raise TypeError("a permutation must be an iterable of integers") from exc

    if any(type(value) is not int for value in result):
        raise TypeError("permutation entries must be built-in integers")

    expected = set(range(1, len(result) + 1))
    if set(result) != expected or len(set(result)) != len(result):
        raise ValueError(
            "a permutation of length n must contain every integer 1, ..., n once"
        )
    return result


def _validate_same_size(
    first: Sequence[int], second: Sequence[int]
) -> tuple[Permutation, Permutation]:
    left = validate_permutation(first)
    right = validate_permutation(second)
    if len(left) != len(right):
        raise ValueError("permutations must have the same size")
    return left, right


def _canonical_cycles_validated(permutation: Permutation) -> tuple[Cycle, ...]:
    visited = [False] * len(permutation)
    cycles: list[Cycle] = []

    # Processing possible starting values in increasing order ensures both
    # that every cycle starts at its minimum and that cycles are sorted by it.
    for start in range(1, len(permutation) + 1):
        if visited[start - 1]:
            continue
        current = start
        cycle: list[int] = []
        while not visited[current - 1]:
            visited[current - 1] = True
            cycle.append(current)
            current = permutation[current - 1]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def canonical_cycles(permutation: Sequence[int]) -> tuple[Cycle, ...]:
    """Return canonical disjoint cycles, including singleton cycles.

    Each cycle starts with its smallest value and follows the permutation
    action.  Cycles are ordered by their first (and therefore smallest) value.
    """

    return _canonical_cycles_validated(validate_permutation(permutation))


def lehmer_code(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return ``L_i = # {j > i : p(j) < p(i)}`` for every position ``i``."""

    p = validate_permutation(permutation)
    seen_to_right = 0
    reversed_code: list[int] = []
    for value in reversed(p):
        smaller_mask = (1 << (value - 1)) - 1
        reversed_code.append((seen_to_right & smaller_mask).bit_count())
        seen_to_right |= 1 << (value - 1)
    return tuple(reversed(reversed_code))


def inversion_vector(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return the value-indexed inversion vector from the Passage convention.

    Entry ``I_v`` (stored at index ``v - 1``) is the number of values larger
    than ``v`` that occur to the left of ``v``.  For example, ``[3, 1, 4, 2]``
    has inversion vector ``(1, 2, 0, 0)``.
    """

    p = validate_permutation(permutation)
    result = [0] * len(p)
    seen_to_left = 0
    for value in p:
        result[value - 1] = (seen_to_left >> value).bit_count()
        seen_to_left |= 1 << (value - 1)
    return tuple(result)


def inversion_count(permutation: Sequence[int]) -> int:
    """Return the number of inversions, equivalently the Coxeter length."""

    p = validate_permutation(permutation)
    seen_to_left = 0
    total = 0
    for value in p:
        total += (seen_to_left >> value).bit_count()
        seen_to_left |= 1 << (value - 1)
    return total


def reduced_coxeter_word(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return a deterministic reduced word in adjacent transpositions.

    To remove ambiguity among multiple reduced words, repeatedly swap the
    *leftmost* descent of ``p`` until reaching the identity, then reverse the
    recorded indices.  If the result is ``(i_1, ..., i_l)``, starting at the
    identity and right multiplying by ``s_{i_1}, ..., s_{i_l}`` reconstructs
    ``p``.  Each ``s_i`` swaps positions ``i`` and ``i + 1`` and indices are
    one-based.
    """

    current = list(validate_permutation(permutation))
    reducing_word: list[int] = []

    while True:
        descent = next(
            (
                index
                for index in range(len(current) - 1)
                if current[index] > current[index + 1]
            ),
            None,
        )
        if descent is None:
            break
        current[descent], current[descent + 1] = (
            current[descent + 1],
            current[descent],
        )
        reducing_word.append(descent + 1)

    return tuple(reversed(reducing_word))


def descent_count(permutation: Sequence[int]) -> int:
    """Return ``# {i : p(i) > p(i + 1)}``."""

    p = validate_permutation(permutation)
    return sum(left > right for left, right in zip(p, p[1:]))


def peak_count(permutation: Sequence[int]) -> int:
    """Return the number of interior peaks.

    A position ``i`` is a peak exactly when
    ``p(i - 1) < p(i) > p(i + 1)``.  Endpoints are never peaks.
    """

    p = validate_permutation(permutation)
    return sum(
        left < center > right
        for left, center, right in zip(p, p[1:], p[2:])
    )


def exceedance_count(permutation: Sequence[int]) -> int:
    """Return ``# {i : p(i) > i}``, with positions numbered from one."""

    p = validate_permutation(permutation)
    return sum(value > index for index, value in enumerate(p, start=1))


def recoil_count(permutation: Sequence[int]) -> int:
    """Return the number of descents of the inverse permutation.

    Equivalently, this counts values ``v`` for which the position of ``v`` is
    greater than the position of ``v + 1``.
    """

    p = validate_permutation(permutation)
    inverse_positions = [0] * len(p)
    for position, value in enumerate(p, start=1):
        inverse_positions[value - 1] = position
    return sum(
        left > right
        for left, right in zip(inverse_positions, inverse_positions[1:])
    )


def fixed_point_count(permutation: Sequence[int]) -> int:
    """Return ``# {i : p(i) = i}``."""

    p = validate_permutation(permutation)
    return sum(value == index for index, value in enumerate(p, start=1))


def parity(permutation: Sequence[int]) -> int:
    """Return ``0`` for an even permutation and ``1`` for an odd one."""

    return inversion_count(permutation) % 2


def cycle_type(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return cycle lengths in descending order, retaining lengths equal to 1."""

    p = validate_permutation(permutation)
    return tuple(
        sorted(
            (len(cycle) for cycle in _canonical_cycles_validated(p)), reverse=True
        )
    )


def rsk_shape(permutation: Sequence[int]) -> tuple[int, ...]:
    """Return the partition shape produced by RSK row insertion."""

    p = validate_permutation(permutation)
    rows: list[list[int]] = []

    for value in p:
        carried = value
        for row in rows:
            insertion_point = bisect_left(row, carried)
            if insertion_point == len(row):
                row.append(carried)
                break
            row[insertion_point], carried = carried, row[insertion_point]
        else:
            rows.append([carried])

    return tuple(len(row) for row in rows)


def _strict_lis_length(values: Sequence[int]) -> int:
    tails: list[int] = []
    for value in values:
        insertion_point = bisect_left(tails, value)
        if insertion_point == len(tails):
            tails.append(value)
        else:
            tails[insertion_point] = value
    return len(tails)


def lis_length(permutation: Sequence[int]) -> int:
    """Return the longest strictly increasing subsequence length."""

    return _strict_lis_length(validate_permutation(permutation))


def lds_length(permutation: Sequence[int]) -> int:
    """Return the longest strictly decreasing subsequence length."""

    p = validate_permutation(permutation)
    return _strict_lis_length(tuple(-value for value in p))


def _standardize(values: Sequence[int]) -> tuple[int, ...]:
    ranks = {value: rank for rank, value in enumerate(sorted(values), start=1)}
    return tuple(ranks[value] for value in values)


def pattern_contains(permutation: Sequence[int], pattern: Sequence[int]) -> bool:
    """Return whether ``permutation`` classically contains ``pattern``.

    Containment means that some subsequence, not necessarily consecutive, has
    the same relative order as ``pattern``.  In accordance with the standard
    combinatorial convention, the empty pattern is contained in every
    permutation.
    """

    p = validate_permutation(permutation)
    target = validate_permutation(pattern)
    pattern_size = len(target)
    if pattern_size > len(p):
        return False

    for indices in combinations(range(len(p)), pattern_size):
        if _standardize(tuple(p[index] for index in indices)) == target:
            return True
    return False


def contains_pattern(permutation: Sequence[int], pattern: Sequence[int]) -> bool:
    """Alias with verb-first naming for :func:`pattern_contains`."""

    return pattern_contains(permutation, pattern)


def avoids_pattern(permutation: Sequence[int], pattern: Sequence[int]) -> bool:
    """Return whether ``permutation`` classically avoids ``pattern``."""

    return not pattern_contains(permutation, pattern)


def _inverse_validated(permutation: Permutation) -> Permutation:
    result = [0] * len(permutation)
    for index, value in enumerate(permutation, start=1):
        result[value - 1] = index
    return tuple(result)


def inverse(permutation: Sequence[int]) -> Permutation:
    """Return the inverse permutation."""

    return _inverse_validated(validate_permutation(permutation))


def _compose_validated(left: Permutation, right: Permutation) -> Permutation:
    return tuple(left[value - 1] for value in right)


def compose(left: Sequence[int], right: Sequence[int]) -> Permutation:
    """Return ``left ∘ right``, so the right operand acts first."""

    a, b = _validate_same_size(left, right)
    return _compose_validated(a, b)


def power(permutation: Sequence[int], exponent: int) -> Permutation:
    """Return a nonnegative integer power using exponentiation by squaring."""

    p = validate_permutation(permutation)
    if type(exponent) is not int:
        raise TypeError("the exponent must be a built-in integer")
    if exponent < 0:
        raise ValueError("the exponent must be nonnegative")

    result = tuple(range(1, len(p) + 1))
    factor = p
    remaining = exponent
    while remaining:
        if remaining & 1:
            result = _compose_validated(result, factor)
        factor = _compose_validated(factor, factor)
        remaining >>= 1
    return result


def conjugate(conjugator: Sequence[int], permutation: Sequence[int]) -> Permutation:
    """Return ``g ∘ p ∘ g^-1`` for arguments ``(g, p)``."""

    g, p = _validate_same_size(conjugator, permutation)
    return _compose_validated(
        _compose_validated(g, p), _inverse_validated(g)
    )


def commutator(first: Sequence[int], second: Sequence[int]) -> Permutation:
    """Return ``a ∘ b ∘ a^-1 ∘ b^-1`` for arguments ``(a, b)``."""

    a, b = _validate_same_size(first, second)
    return _compose_validated(
        _compose_validated(
            _compose_validated(a, b), _inverse_validated(a)
        ),
        _inverse_validated(b),
    )


def right_multiply_simple(permutation: Sequence[int], index: int) -> Permutation:
    """Return ``p ∘ s_i`` by swapping positions ``i`` and ``i + 1``.

    ``index`` is one-based and must satisfy ``1 <= index < n``.
    """

    p = validate_permutation(permutation)
    if type(index) is not int:
        raise TypeError("the simple-transposition index must be a built-in integer")
    if not 1 <= index < len(p):
        raise ValueError("the simple-transposition index must satisfy 1 <= i < n")

    result = list(p)
    result[index - 1], result[index] = result[index], result[index - 1]
    return tuple(result)


def bruhat_leq(first: Sequence[int], second: Sequence[int]) -> bool:
    """Return whether ``first <= second`` in strong Bruhat order.

    This uses the rank-matrix criterion.  With
    ``r_w(p, q) = # {i <= p : w(i) <= q}``, one has ``u <= v`` exactly when
    ``r_u(p, q) >= r_v(p, q)`` for every ``p, q``.  Equality is therefore
    included.
    """

    u, v = _validate_same_size(first, second)
    size = len(u)
    u_prefix_counts = [0] * (size + 1)
    v_prefix_counts = [0] * (size + 1)

    for u_value, v_value in zip(u, v):
        u_prefix_counts[u_value] += 1
        v_prefix_counts[v_value] += 1

        u_rank = 0
        v_rank = 0
        for threshold in range(1, size + 1):
            u_rank += u_prefix_counts[threshold]
            v_rank += v_prefix_counts[threshold]
            if u_rank < v_rank:
                return False
    return True

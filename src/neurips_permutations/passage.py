"""Passage Math tokenization and rendering for permutation examples.

The core grammar follows the supplied permutation-tokenizer specification:

    <BOS> <SIZE> NUMBER <ONE_START> ... <ONE_END>
        OPERANDS TASK = ANSWER <EOS>

Numbers are non-negative base-100 integers.  Values below 100 are represented
by exactly one two-digit token; larger values are enclosed by ``<NUM_START>``
and ``<NUM_END>`` and written most-significant base-100 digit first.

This module deliberately renders tokens rather than relying on punctuation in
free-form text.  ``render_passage`` is the convenient string interface and
``passage_tokens`` exposes the exact atomic-token sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence


NUMBER_TOKENS: tuple[str, ...] = tuple(f"{value:02d}" for value in range(100))

# Keep the original fixed tokens first, then append the extension needed by
# the twenty-task dataset.  Punctuation is an atomic token too.
ORIGINAL_FIXED_TOKENS: tuple[str, ...] = (
    "<NUM_START>",
    "<NUM_END>",
    "<BOS>",
    "<EOS>",
    "<PAD>",
    "<ONE_START>",
    "<ONE_END>",
    "<CYCLE_START>",
    "<CYCLE_END>",
    "<LEHMER_START>",
    "<LEHMER_END>",
    "<INVEC_START>",
    "<INVEC_END>",
    ",",
    ";",
    "<LENGTH>",
    "<PARITY>",
    "<PEAKS>",
    "<EXCEEDANCES>",
    "<FIXED_POINTS>",
    "<DESCENTS>",
    "<RECOILS>",
    "<LIS_LENGTH>",
    "<SIZE>",
    "=",
    "+",
    "-",
    "*",
    "/",
    "<YT_ROW_START>",
    "<YT_ROW_END>",
    "<YT_COL_START>",
    "<YT_COL_END>",
    "|",
    "<P_TABLEAU>",
    "<Q_TABLEAU>",
)

EXTENDED_FIXED_TOKENS: tuple[str, ...] = (
    "<TO_CYCLE>",
    "<TO_LEHMER>",
    "<TO_INVERSION_VECTOR>",
    "<TO_REDUCED_WORD>",
    "<CYCLE_TYPE>",
    "<RSK_SHAPE>",
    "<LDS_LENGTH>",
    "<PATTERN_AVOIDANCE>",
    "<INVERSE>",
    "<COMPOSE>",
    "<POWER>",
    "<CONJUGATE>",
    "<COMMUTATOR>",
    "<RIGHT_MULTIPLY_SIMPLE>",
    "<BRUHAT_LEQ>",
    "<ARG_START>",
    "<ARG_END>",
    "<PATTERN_START>",
    "<PATTERN_END>",
    "<EXPONENT>",
    "<SIMPLE_INDEX>",
    "<REDUCED_WORD_START>",
    "<REDUCED_WORD_END>",
    "<CYCLE_TYPE_START>",
    "<CYCLE_TYPE_END>",
    "<RSK_SHAPE_START>",
    "<RSK_SHAPE_END>",
)

PROPERTY_FIXED_TOKENS: tuple[str, ...] = (
    "<VALLEYS>",
    "<DOUBLE_ASCENTS>",
    "<DOUBLE_DESCENTS>",
    "<SUCCESSIONS>",
    "<ADJACENCIES>",
    "<ANTI_FIXED_POINTS>",
    "<DEFICIENCIES>",
    "<LEFT_TO_RIGHT_MAXIMA>",
    "<LEFT_TO_RIGHT_MINIMA>",
    "<RIGHT_TO_LEFT_MAXIMA>",
    "<RIGHT_TO_LEFT_MINIMA>",
    "<CYCLE_COUNT>",
    "<TWO_CYCLE_COUNT>",
    "<THREE_CYCLE_COUNT>",
    "<EVEN_CYCLE_COUNT>",
    "<ODD_CYCLE_COUNT>",
    "<LONGEST_CYCLE>",
    "<SHORTEST_CYCLE>",
    "<NONTRIVIAL_CYCLE_COUNT>",
    "<LONGEST_INCREASING_RUN>",
    "<LONGEST_DECREASING_RUN>",
    "<GLOBAL_DESCENTS>",
    "<COMPONENTS>",
    "<MAX_DISPLACEMENT>",
    "<DISPLACEMENT_ONE_COUNT>",
)

FIXED_TOKENS: tuple[str, ...] = (
    ORIGINAL_FIXED_TOKENS + EXTENDED_FIXED_TOKENS + PROPERTY_FIXED_TOKENS
)
VOCABULARY: tuple[str, ...] = NUMBER_TOKENS + FIXED_TOKENS
TOKEN_TO_ID: Mapping[str, int] = {token: index for index, token in enumerate(VOCABULARY)}
ID_TO_TOKEN: Mapping[int, str] = {index: token for token, index in TOKEN_TO_ID.items()}


@dataclass(frozen=True)
class TaskSpec:
    """The fixed task token, typed operand, and typed answer for one task."""

    token: str
    operand_kind: str | None
    answer_kind: str


TASK_SPECS: Mapping[str, TaskSpec] = {
    "to_cycle": TaskSpec("<TO_CYCLE>", None, "cycle"),
    "to_lehmer": TaskSpec("<TO_LEHMER>", None, "lehmer"),
    "to_inversion_vector": TaskSpec(
        "<TO_INVERSION_VECTOR>", None, "inversion_vector"
    ),
    "to_reduced_word": TaskSpec("<TO_REDUCED_WORD>", None, "reduced_word"),
    "length": TaskSpec("<LENGTH>", None, "scalar"),
    "descents": TaskSpec("<DESCENTS>", None, "scalar"),
    "fixed_points": TaskSpec("<FIXED_POINTS>", None, "scalar"),
    "parity": TaskSpec("<PARITY>", None, "boolean"),
    "cycle_type": TaskSpec("<CYCLE_TYPE>", None, "cycle_type"),
    "rsk_shape": TaskSpec("<RSK_SHAPE>", None, "rsk_shape"),
    "lis_length": TaskSpec("<LIS_LENGTH>", None, "scalar"),
    "lds_length": TaskSpec("<LDS_LENGTH>", None, "scalar"),
    "pattern_avoidance": TaskSpec(
        "<PATTERN_AVOIDANCE>", "pattern", "boolean"
    ),
    "peaks": TaskSpec("<PEAKS>", None, "scalar"),
    "exceedances": TaskSpec("<EXCEEDANCES>", None, "scalar"),
    "recoils": TaskSpec("<RECOILS>", None, "scalar"),
    "inverse": TaskSpec("<INVERSE>", None, "permutation"),
    "compose": TaskSpec("<COMPOSE>", "operand", "permutation"),
    "power": TaskSpec("<POWER>", "exponent", "permutation"),
    "conjugate": TaskSpec("<CONJUGATE>", "operand", "permutation"),
    "commutator": TaskSpec("<COMMUTATOR>", "operand", "permutation"),
    "right_multiply_simple": TaskSpec(
        "<RIGHT_MULTIPLY_SIMPLE>", "simple_index", "permutation"
    ),
    "bruhat_leq": TaskSpec("<BRUHAT_LEQ>", "operand", "boolean"),
    "valleys": TaskSpec("<VALLEYS>", None, "scalar"),
    "double_ascents": TaskSpec("<DOUBLE_ASCENTS>", None, "scalar"),
    "double_descents": TaskSpec("<DOUBLE_DESCENTS>", None, "scalar"),
    "successions": TaskSpec("<SUCCESSIONS>", None, "scalar"),
    "adjacencies": TaskSpec("<ADJACENCIES>", None, "scalar"),
    "anti_fixed_points": TaskSpec("<ANTI_FIXED_POINTS>", None, "scalar"),
    "deficiencies": TaskSpec("<DEFICIENCIES>", None, "scalar"),
    "left_to_right_maxima": TaskSpec(
        "<LEFT_TO_RIGHT_MAXIMA>", None, "scalar"
    ),
    "left_to_right_minima": TaskSpec(
        "<LEFT_TO_RIGHT_MINIMA>", None, "scalar"
    ),
    "right_to_left_maxima": TaskSpec(
        "<RIGHT_TO_LEFT_MAXIMA>", None, "scalar"
    ),
    "right_to_left_minima": TaskSpec(
        "<RIGHT_TO_LEFT_MINIMA>", None, "scalar"
    ),
    "cycle_count": TaskSpec("<CYCLE_COUNT>", None, "scalar"),
    "two_cycle_count": TaskSpec("<TWO_CYCLE_COUNT>", None, "scalar"),
    "three_cycle_count": TaskSpec("<THREE_CYCLE_COUNT>", None, "scalar"),
    "even_cycle_count": TaskSpec("<EVEN_CYCLE_COUNT>", None, "scalar"),
    "odd_cycle_count": TaskSpec("<ODD_CYCLE_COUNT>", None, "scalar"),
    "longest_cycle": TaskSpec("<LONGEST_CYCLE>", None, "scalar"),
    "shortest_cycle": TaskSpec("<SHORTEST_CYCLE>", None, "scalar"),
    "nontrivial_cycle_count": TaskSpec(
        "<NONTRIVIAL_CYCLE_COUNT>", None, "scalar"
    ),
    "longest_increasing_run": TaskSpec(
        "<LONGEST_INCREASING_RUN>", None, "scalar"
    ),
    "longest_decreasing_run": TaskSpec(
        "<LONGEST_DECREASING_RUN>", None, "scalar"
    ),
    "global_descents": TaskSpec("<GLOBAL_DESCENTS>", None, "scalar"),
    "components": TaskSpec("<COMPONENTS>", None, "scalar"),
    "max_displacement": TaskSpec("<MAX_DISPLACEMENT>", None, "scalar"),
    "displacement_one_count": TaskSpec(
        "<DISPLACEMENT_ONE_COUNT>", None, "scalar"
    ),
}


_LEXEME_RE = re.compile(r"<[A-Z][A-Z0-9_]*>|[0-9]{2}|[,;=+*/|\-]")


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def encode_number(value: int) -> tuple[str, ...]:
    """Return the unique atomic-token encoding of a non-negative integer."""

    value = _require_int(value, name="value")
    if value < 100:
        return (f"{value:02d}",)

    digits: list[int] = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining, 100)
        digits.append(digit)
    digits.reverse()
    return (
        "<NUM_START>",
        *(f"{digit:02d}" for digit in digits),
        "<NUM_END>",
    )


# Descriptive alias used by callers that want to emphasize the domain.
encode_nonnegative = encode_number


def decode_number(tokens: Sequence[str], start: int = 0) -> tuple[int, int]:
    """Decode one canonical number, returning ``(value, next_token_index)``."""

    start = _require_int(start, name="start")
    if start >= len(tokens):
        raise ValueError("no number token at the requested start index")

    token = tokens[start]
    if token in NUMBER_TOKENS:
        return int(token), start + 1
    if token != "<NUM_START>":
        raise ValueError(f"expected a number token, got {token!r}")

    end = start + 1
    digits: list[int] = []
    while end < len(tokens) and tokens[end] != "<NUM_END>":
        digit = tokens[end]
        if digit not in NUMBER_TOKENS:
            raise ValueError(f"invalid base-100 digit {digit!r}")
        digits.append(int(digit))
        end += 1
    if end >= len(tokens):
        raise ValueError("unterminated <NUM_START> block")
    if len(digits) < 2 or digits[0] == 0:
        raise ValueError("non-canonical wrapped number")

    value = 0
    for digit in digits:
        value = value * 100 + digit
    if value < 100:
        raise ValueError("values below 100 must use one atomic number token")
    return value, end + 1


def tokenize(text: str) -> tuple[str, ...]:
    """Lex and validate Passage Math text into its atomic vocabulary tokens.

    Whitespace is insignificant, so the tokenizer also accepts compact input
    such as ``<ONE_START>03,01<ONE_END>``.  Rendered output always puts one
    space between tokens for readability.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens: list[str] = []
    cursor = 0
    for match in _LEXEME_RE.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError(f"unrecognized Passage Math text at offset {cursor}")
        token = match.group(0)
        if token not in TOKEN_TO_ID:
            raise ValueError(f"unknown Passage Math token {token!r}")
        tokens.append(token)
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError(f"unrecognized Passage Math text at offset {cursor}")
    return tuple(tokens)


def token_ids(tokens_or_text: Sequence[str] | str) -> tuple[int, ...]:
    """Map a token sequence (or Passage Math string) to stable vocabulary IDs."""

    tokens = tokenize(tokens_or_text) if isinstance(tokens_or_text, str) else tokens_or_text
    try:
        return tuple(TOKEN_TO_ID[token] for token in tokens)
    except KeyError as error:
        raise ValueError(f"unknown Passage Math token {error.args[0]!r}") from None


def _as_int_tuple(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    for index, value in enumerate(result):
        _require_int(value, name=f"{name}[{index}]")
    return result


def validate_permutation(
    permutation: Iterable[int], *, name: str = "permutation"
) -> tuple[int, ...]:
    """Return a tuple after checking that it permutes ``{1, ..., n}``."""

    values = _as_int_tuple(permutation, name=name)
    if not values:
        raise ValueError(f"{name} must contain at least one entry")
    expected = list(range(1, len(values) + 1))
    if sorted(values) != expected:
        raise ValueError(f"{name} must be a permutation of 1..{len(values)}")
    return values


def _list_body(values: Iterable[int], *, name: str) -> tuple[str, ...]:
    items = _as_int_tuple(values, name=name)
    tokens: list[str] = []
    for index, item in enumerate(items):
        if index:
            tokens.append(",")
        tokens.extend(encode_number(item))
    return tuple(tokens)


def _bounded_list(
    start_token: str, values: Iterable[int], end_token: str, *, name: str
) -> tuple[str, ...]:
    return (start_token, *_list_body(values, name=name), end_token)


def one_line_tokens(permutation: Iterable[int]) -> tuple[str, ...]:
    values = validate_permutation(permutation)
    return _bounded_list("<ONE_START>", values, "<ONE_END>", name="permutation")


def canonical_cycles(permutation: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Return cycles in the attachment's required canonical order."""

    values = validate_permutation(permutation)
    visited: set[int] = set()
    cycles: list[tuple[int, ...]] = []
    for start in range(1, len(values) + 1):
        if start in visited:
            continue
        cycle: list[int] = []
        current = start
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            current = values[current - 1]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def _cycle_block(cycles: Sequence[Sequence[int]], *, n: int) -> tuple[str, ...]:
    if not cycles:
        raise ValueError("cycles must include singleton cycles")

    normalized: list[tuple[int, ...]] = []
    for cycle_index, cycle in enumerate(cycles):
        values = _as_int_tuple(cycle, name=f"cycles[{cycle_index}]")
        if not values:
            raise ValueError("cycles cannot contain an empty cycle")
        if values[0] != min(values):
            raise ValueError("each cycle must start with its smallest element")
        normalized.append(values)
    if [cycle[0] for cycle in normalized] != sorted(cycle[0] for cycle in normalized):
        raise ValueError("cycles must be ordered by their smallest elements")
    flattened = [value for cycle in normalized for value in cycle]
    if sorted(flattened) != list(range(1, n + 1)):
        raise ValueError("canonical cycles must contain every value 1..n exactly once")

    tokens: list[str] = ["<CYCLE_START>"]
    for cycle_index, cycle in enumerate(normalized):
        if cycle_index:
            tokens.append(";")
        tokens.extend(_list_body(cycle, name=f"cycles[{cycle_index}]"))
    tokens.append("<CYCLE_END>")
    return tuple(tokens)


def cycle_tokens(permutation: Iterable[int]) -> tuple[str, ...]:
    values = validate_permutation(permutation)
    return _cycle_block(canonical_cycles(values), n=len(values))


def lehmer_code(permutation: Iterable[int]) -> tuple[int, ...]:
    values = validate_permutation(permutation)
    return tuple(
        sum(right < value for right in values[index + 1 :])
        for index, value in enumerate(values)
    )


def lehmer_tokens(permutation: Iterable[int]) -> tuple[str, ...]:
    return _bounded_list(
        "<LEHMER_START>",
        lehmer_code(permutation),
        "<LEHMER_END>",
        name="lehmer_code",
    )


def inversion_vector(permutation: Iterable[int]) -> tuple[int, ...]:
    values = validate_permutation(permutation)
    positions = {value: index for index, value in enumerate(values)}
    return tuple(
        sum(positions[larger] < positions[value] for larger in range(value + 1, len(values) + 1))
        for value in range(1, len(values) + 1)
    )


def inversion_vector_tokens(permutation: Iterable[int]) -> tuple[str, ...]:
    return _bounded_list(
        "<INVEC_START>",
        inversion_vector(permutation),
        "<INVEC_END>",
        name="inversion_vector",
    )


def _partition_answer(
    answer: Iterable[int], *, n: int, start: str, end: str, name: str
) -> tuple[str, ...]:
    parts = _as_int_tuple(answer, name=name)
    if not parts or any(part == 0 for part in parts):
        raise ValueError(f"{name} must be a nonempty sequence of positive parts")
    if any(left < right for left, right in zip(parts, parts[1:])):
        raise ValueError(f"{name} must be in non-increasing canonical order")
    if sum(parts) != n:
        raise ValueError(f"{name} parts must sum to permutation size {n}")
    return _bounded_list(start, parts, end, name=name)


def _boolean_answer(answer: object) -> tuple[str, ...]:
    if isinstance(answer, bool):
        value = int(answer)
    elif isinstance(answer, int) and answer in (0, 1):
        value = answer
    else:
        raise ValueError("boolean and parity answers must be bool, 0, or 1")
    return encode_number(value)


def _cycles_to_permutation(cycles: Sequence[Sequence[int]], *, n: int) -> tuple[int, ...]:
    result = [0] * n
    for cycle in cycles:
        for index, value in enumerate(cycle):
            result[value - 1] = cycle[(index + 1) % len(cycle)]
    return tuple(result)


def _answer_tokens(
    spec: TaskSpec,
    answer: object,
    *,
    primary: tuple[int, ...],
) -> tuple[str, ...]:
    n = len(primary)
    kind = spec.answer_kind
    if kind == "scalar":
        return encode_number(_require_int(answer, name="answer"))
    if kind == "boolean":
        return _boolean_answer(answer)
    if kind == "permutation":
        values = validate_permutation(answer, name="answer")  # type: ignore[arg-type]
        if len(values) != n:
            raise ValueError("permutation answer must have the primary permutation's size")
        return one_line_tokens(values)
    if kind == "cycle":
        # Accept either the source permutation (convenient for generators) or
        # explicit canonical cycles (convenient when loading materialized data).
        raw = tuple(answer)  # type: ignore[arg-type]
        if raw and all(isinstance(value, int) and not isinstance(value, bool) for value in raw):
            values = validate_permutation(raw, name="answer")
            if values != primary:
                raise ValueError("to_cycle answer must represent the primary permutation")
            return cycle_tokens(values)
        cycles = tuple(tuple(cycle) for cycle in raw)  # type: ignore[arg-type]
        block = _cycle_block(cycles, n=n)
        if _cycles_to_permutation(cycles, n=n) != primary:
            raise ValueError("to_cycle answer must represent the primary permutation")
        return block
    if kind == "lehmer":
        values = _as_int_tuple(answer, name="answer")  # type: ignore[arg-type]
        if values != lehmer_code(primary):
            raise ValueError("to_lehmer answer does not match the primary permutation")
        return _bounded_list("<LEHMER_START>", values, "<LEHMER_END>", name="answer")
    if kind == "inversion_vector":
        values = _as_int_tuple(answer, name="answer")  # type: ignore[arg-type]
        if values != inversion_vector(primary):
            raise ValueError(
                "to_inversion_vector answer does not match the primary permutation"
            )
        return _bounded_list("<INVEC_START>", values, "<INVEC_END>", name="answer")
    if kind == "reduced_word":
        values = _as_int_tuple(answer, name="answer")  # type: ignore[arg-type]
        if any(value < 1 or value >= n for value in values):
            raise ValueError("reduced-word indices must satisfy 1 <= i < n")
        return _bounded_list(
            "<REDUCED_WORD_START>", values, "<REDUCED_WORD_END>", name="answer"
        )
    if kind == "cycle_type":
        return _partition_answer(
            answer,  # type: ignore[arg-type]
            n=n,
            start="<CYCLE_TYPE_START>",
            end="<CYCLE_TYPE_END>",
            name="cycle_type",
        )
    if kind == "rsk_shape":
        return _partition_answer(
            answer,  # type: ignore[arg-type]
            n=n,
            start="<RSK_SHAPE_START>",
            end="<RSK_SHAPE_END>",
            name="rsk_shape",
        )
    raise AssertionError(f"unhandled answer kind {kind!r}")


def _operand_tokens(
    spec: TaskSpec,
    *,
    n: int,
    operand: Iterable[int] | None,
    pattern: Iterable[int] | None,
    exponent: int | None,
    simple_index: int | None,
) -> tuple[str, ...]:
    supplied = {
        "operand": operand,
        "pattern": pattern,
        "exponent": exponent,
        "simple_index": simple_index,
    }
    expected = spec.operand_kind
    unexpected = [name for name, value in supplied.items() if value is not None and name != expected]
    if unexpected:
        raise ValueError(f"unexpected operand(s) for task: {', '.join(unexpected)}")
    if expected is None:
        return ()
    if supplied[expected] is None:
        raise ValueError(f"task requires the {expected} operand")

    if expected == "operand":
        values = validate_permutation(operand, name="operand")  # type: ignore[arg-type]
        if len(values) != n:
            raise ValueError("binary permutation operand must match the primary size")
        return _bounded_list("<ARG_START>", values, "<ARG_END>", name="operand")
    if expected == "pattern":
        values = validate_permutation(pattern, name="pattern")  # type: ignore[arg-type]
        if len(values) > n:
            raise ValueError("pattern cannot be longer than the primary permutation")
        return _bounded_list(
            "<PATTERN_START>", values, "<PATTERN_END>", name="pattern"
        )
    if expected == "exponent":
        value = _require_int(exponent, name="exponent")
        return ("<EXPONENT>", *encode_number(value))
    if expected == "simple_index":
        value = _require_int(simple_index, name="simple_index", minimum=1)
        if value >= n:
            raise ValueError("simple_index must satisfy 1 <= i < n")
        return ("<SIMPLE_INDEX>", *encode_number(value))
    raise AssertionError(f"unhandled operand kind {expected!r}")


def passage_tokens(
    task: str,
    primary: Iterable[int],
    answer: object,
    *,
    operand: Iterable[int] | None = None,
    pattern: Iterable[int] | None = None,
    exponent: int | None = None,
    simple_index: int | None = None,
) -> tuple[str, ...]:
    """Build one validated Passage Math sequence for any supported task."""

    try:
        spec = TASK_SPECS[task]
    except KeyError:
        valid = ", ".join(TASK_SPECS)
        raise ValueError(f"unknown task {task!r}; expected one of: {valid}") from None

    primary_values = validate_permutation(primary, name="primary")
    operands = _operand_tokens(
        spec,
        n=len(primary_values),
        operand=operand,
        pattern=pattern,
        exponent=exponent,
        simple_index=simple_index,
    )
    answer_block = _answer_tokens(spec, answer, primary=primary_values)
    return (
        "<BOS>",
        "<SIZE>",
        *encode_number(len(primary_values)),
        *one_line_tokens(primary_values),
        *operands,
        spec.token,
        "=",
        *answer_block,
        "<EOS>",
    )


def render_passage(
    task: str,
    primary: Iterable[int],
    answer: object,
    *,
    operand: Iterable[int] | None = None,
    pattern: Iterable[int] | None = None,
    exponent: int | None = None,
    simple_index: int | None = None,
) -> str:
    """Render one Passage Math sequence with exactly one space per token."""

    return " ".join(
        passage_tokens(
            task,
            primary,
            answer,
            operand=operand,
            pattern=pattern,
            exponent=exponent,
            simple_index=simple_index,
        )
    )


__all__ = [
    "EXTENDED_FIXED_TOKENS",
    "FIXED_TOKENS",
    "ID_TO_TOKEN",
    "NUMBER_TOKENS",
    "ORIGINAL_FIXED_TOKENS",
    "PROPERTY_FIXED_TOKENS",
    "TASK_SPECS",
    "TOKEN_TO_ID",
    "TaskSpec",
    "VOCABULARY",
    "canonical_cycles",
    "cycle_tokens",
    "decode_number",
    "encode_nonnegative",
    "encode_number",
    "inversion_vector",
    "inversion_vector_tokens",
    "lehmer_code",
    "lehmer_tokens",
    "one_line_tokens",
    "passage_tokens",
    "render_passage",
    "token_ids",
    "tokenize",
    "validate_permutation",
]

from __future__ import annotations

import pytest

from neurips_permutations.passage import (
    NUMBER_TOKENS,
    ORIGINAL_FIXED_TOKENS,
    TASK_SPECS,
    canonical_cycles,
    cycle_tokens,
    decode_number,
    encode_number,
    inversion_vector,
    inversion_vector_tokens,
    lehmer_code,
    lehmer_tokens,
    passage_tokens,
    render_passage,
    token_ids,
    tokenize,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, ("00",)),
        (7, ("07",)),
        (28, ("28",)),
        (83, ("83",)),
        (99, ("99",)),
        (100, ("<NUM_START>", "01", "00", "<NUM_END>")),
        (137, ("<NUM_START>", "01", "37", "<NUM_END>")),
        (999, ("<NUM_START>", "09", "99", "<NUM_END>")),
        (9999, ("<NUM_START>", "99", "99", "<NUM_END>")),
        (10000, ("<NUM_START>", "01", "00", "00", "<NUM_END>")),
    ],
)
def test_attachment_number_encoding(value: int, expected: tuple[str, ...]) -> None:
    assert encode_number(value) == expected
    assert decode_number(expected) == (value, len(expected))


def test_number_encoding_is_canonical_and_nonnegative() -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        decode_number(("<NUM_START>", "00", "07", "<NUM_END>"))
    with pytest.raises(ValueError, match="non-canonical"):
        decode_number(("<NUM_START>", "07", "<NUM_END>"))
    with pytest.raises(ValueError, match="at least 0"):
        encode_number(-1)
    with pytest.raises(TypeError, match="integer"):
        encode_number(True)


def test_original_attachment_vocabulary_has_100_plus_36_tokens() -> None:
    assert NUMBER_TOKENS == tuple(f"{value:02d}" for value in range(100))
    assert len(ORIGINAL_FIXED_TOKENS) == 36


def test_attachment_representation_fixtures() -> None:
    permutation = (3, 1, 4, 2)
    assert canonical_cycles(permutation) == ((1, 3, 4, 2),)
    assert cycle_tokens(permutation) == (
        "<CYCLE_START>",
        "01",
        ",",
        "03",
        ",",
        "04",
        ",",
        "02",
        "<CYCLE_END>",
    )
    assert lehmer_code(permutation) == (2, 0, 1, 0)
    assert lehmer_tokens(permutation) == (
        "<LEHMER_START>",
        "02",
        ",",
        "00",
        ",",
        "01",
        ",",
        "00",
        "<LEHMER_END>",
    )
    assert inversion_vector(permutation) == (1, 2, 0, 0)
    assert inversion_vector_tokens(permutation) == (
        "<INVEC_START>",
        "01",
        ",",
        "02",
        ",",
        "00",
        ",",
        "00",
        "<INVEC_END>",
    )


def test_cycle_notation_includes_singletons_and_sorts_cycles() -> None:
    assert canonical_cycles((1, 4, 3, 2)) == ((1,), (2, 4), (3,))
    assert " ".join(cycle_tokens((1, 4, 3, 2))) == (
        "<CYCLE_START> 01 ; 02 , 04 ; 03 <CYCLE_END>"
    )


def test_exact_attachment_training_sequence() -> None:
    assert render_passage("descents", (3, 1, 4, 2), 2) == (
        "<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END> "
        "<DESCENTS> = 02 <EOS>"
    )


@pytest.mark.parametrize(
    ("task", "answer", "task_token", "answer_token"),
    [
        ("peaks", 1, "<PEAKS>", "01"),
        ("exceedances", 2, "<EXCEEDANCES>", "02"),
        ("recoils", 1, "<RECOILS>", "01"),
    ],
)
def test_new_statistics_use_original_passage_tokens(
    task: str, answer: int, task_token: str, answer_token: str
) -> None:
    assert render_passage(task, (3, 1, 4, 2), answer) == (
        "<BOS> <SIZE> 04 <ONE_START> 03 , 01 , 04 , 02 <ONE_END> "
        f"{task_token} = {answer_token} <EOS>"
    )


@pytest.mark.parametrize("task", ["peaks", "exceedances", "recoils"])
def test_new_statistics_accept_zero_without_a_wrapped_number(task: str) -> None:
    tokens = passage_tokens(task, (1,), 0)
    assert tokens[-4:] == (TASK_SPECS[task].token, "=", "00", "<EOS>")


def test_tokenizer_handles_readable_and_compact_passage_math() -> None:
    readable = render_passage("descents", (3, 1, 4, 2), 2)
    compact = readable.replace(" ", "")
    assert tokenize(compact) == passage_tokens("descents", (3, 1, 4, 2), 2)
    assert token_ids(readable) == token_ids(tokenize(readable))
    with pytest.raises(ValueError, match="unknown"):
        tokenize("<NOT_A_TOKEN>")


def test_binary_operand_is_a_typed_one_line_payload() -> None:
    tokens = passage_tokens(
        "compose",
        (2, 1, 3),
        (1, 3, 2),
        operand=(3, 2, 1),
    )
    expected_subsequence = (
        "<ARG_START>",
        "03",
        ",",
        "02",
        ",",
        "01",
        "<ARG_END>",
        "<COMPOSE>",
        "=",
        "<ONE_START>",
        "01",
        ",",
        "03",
        ",",
        "02",
        "<ONE_END>",
    )
    start = tokens.index("<ARG_START>")
    assert tokens[start : start + len(expected_subsequence)] == expected_subsequence


def test_pattern_exponent_and_simple_index_operands() -> None:
    pattern = passage_tokens(
        "pattern_avoidance",
        (3, 1, 4, 2),
        True,
        pattern=(2, 1, 3),
    )
    pattern_start = pattern.index("<PATTERN_START>")
    expected_pattern_block = (
        "<PATTERN_START>",
        "02",
        ",",
        "01",
        ",",
        "03",
        "<PATTERN_END>",
        "<PATTERN_AVOIDANCE>",
        "=",
        "01",
    )
    assert (
        pattern[pattern_start : pattern_start + len(expected_pattern_block)]
        == expected_pattern_block
    )

    power = passage_tokens("power", (2, 1), (1, 2), exponent=100)
    exponent_start = power.index("<EXPONENT>")
    assert power[exponent_start : exponent_start + 5] == (
        "<EXPONENT>",
        "<NUM_START>",
        "01",
        "00",
        "<NUM_END>",
    )

    simple = passage_tokens(
        "right_multiply_simple", (2, 3, 1), (3, 2, 1), simple_index=2
    )
    simple_start = simple.index("<SIMPLE_INDEX>")
    assert simple[simple_start : simple_start + 2] == ("<SIMPLE_INDEX>", "02")


@pytest.mark.parametrize(
    ("task", "answer", "boundaries"),
    [
        ("to_cycle", (3, 1, 4, 2), ("<CYCLE_START>", "<CYCLE_END>")),
        ("to_lehmer", (2, 0, 1, 0), ("<LEHMER_START>", "<LEHMER_END>")),
        (
            "to_inversion_vector",
            (1, 2, 0, 0),
            ("<INVEC_START>", "<INVEC_END>"),
        ),
        (
            "to_reduced_word",
            (2, 1, 3),
            ("<REDUCED_WORD_START>", "<REDUCED_WORD_END>"),
        ),
        (
            "cycle_type",
            (4,),
            ("<CYCLE_TYPE_START>", "<CYCLE_TYPE_END>"),
        ),
        ("rsk_shape", (2, 2), ("<RSK_SHAPE_START>", "<RSK_SHAPE_END>")),
        ("inverse", (2, 4, 1, 3), ("<ONE_START>", "<ONE_END>")),
    ],
)
def test_structured_answers_have_task_specific_boundaries(
    task: str, answer: object, boundaries: tuple[str, str]
) -> None:
    tokens = passage_tokens(task, (3, 1, 4, 2), answer)
    equals = tokens.index("=")
    assert tokens[equals + 1] == boundaries[0]
    assert tokens[-2] == boundaries[1]


@pytest.mark.parametrize("task", ["parity", "pattern_avoidance", "bruhat_leq"])
def test_boolean_tasks_use_00_or_01(task: str) -> None:
    kwargs: dict[str, object] = {}
    if task == "pattern_avoidance":
        kwargs["pattern"] = (2, 1)
    elif task == "bruhat_leq":
        kwargs["operand"] = (2, 1, 3)
    assert passage_tokens(task, (2, 1, 3), True, **kwargs)[-2] == "01"
    assert passage_tokens(task, (2, 1, 3), 0, **kwargs)[-2] == "00"


def test_all_v2_and_v3_tasks_have_unique_task_tokens() -> None:
    assert len(TASK_SPECS) == 23
    assert len({spec.token for spec in TASK_SPECS.values()}) == 23
    assert {
        task: (TASK_SPECS[task].token, TASK_SPECS[task].answer_kind)
        for task in ("peaks", "exceedances", "recoils")
    } == {
        "peaks": ("<PEAKS>", "scalar"),
        "exceedances": ("<EXCEEDANCES>", "scalar"),
        "recoils": ("<RECOILS>", "scalar"),
    }


def test_missing_or_extraneous_typed_operands_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires the operand"):
        passage_tokens("compose", (2, 1), (2, 1))
    with pytest.raises(ValueError, match="unexpected operand"):
        passage_tokens("inverse", (2, 1), (2, 1), exponent=2)
    with pytest.raises(ValueError, match="match the primary size"):
        passage_tokens("bruhat_leq", (2, 1), True, operand=(1,))


def test_partition_answers_are_canonical() -> None:
    with pytest.raises(ValueError, match="non-increasing"):
        passage_tokens("rsk_shape", (3, 1, 4, 2), (1, 3))
    with pytest.raises(ValueError, match="sum"):
        passage_tokens("cycle_type", (3, 1, 4, 2), (3,))

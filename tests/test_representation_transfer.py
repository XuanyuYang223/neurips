from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from neurips_permutations import math_ops as ops
from neurips_permutations.passage import passage_tokens
from neurips_permutations.representation_transfer import (
    BASE_TASKS,
    FULL_COMBINATIONS,
    HELD_OUT_COMBINATIONS,
    TRAIN_COMBINATIONS,
    _record,
    combinations_for_split,
    summarize_cells,
    verify_manifest,
)


def test_frozen_row_plus_column_grid() -> None:
    assert len(BASE_TASKS) == 8
    assert len(FULL_COMBINATIONS) == 32
    assert len(TRAIN_COMBINATIONS) == 11
    assert len(HELD_OUT_COMBINATIONS) == 21
    assert set(TRAIN_COMBINATIONS).isdisjoint(HELD_OUT_COMBINATIONS)
    assert set(TRAIN_COMBINATIONS) | set(HELD_OUT_COMBINATIONS) == set(
        FULL_COMBINATIONS
    )
    assert combinations_for_split("train") == TRAIN_COMBINATIONS
    assert combinations_for_split("validation") == FULL_COMBINATIONS


@pytest.mark.parametrize(
    "representation,start,end",
    [
        ("one_line", "<ONE_START>", "<ONE_END>"),
        ("cycle", "<CYCLE_START>", "<CYCLE_END>"),
        ("lehmer", "<LEHMER_START>", "<LEHMER_END>"),
        ("inversion_vector", "<INVEC_START>", "<INVEC_END>"),
    ],
)
def test_passage_renders_each_input_representation(
    representation: str, start: str, end: str
) -> None:
    tokens = passage_tokens(
        "descents", (3, 1, 4, 2), 2, representation=representation
    )
    assert start in tokens and end in tokens
    assert tokens[-4:] == ("<DESCENTS>", "=", "02", "<EOS>")


def test_derived_record_recomputes_truth() -> None:
    source = {
        "id": 7,
        "task": "length",
        "inputs": {"primary": [3, 1, 4, 2]},
        "answer": 999,
    }
    record = _record(source, "lehmer")
    assert record["answer"] == ops.inversion_count((3, 1, 4, 2)) == 3
    assert record["id"] == 7 * 32 + FULL_COMBINATIONS.index("lehmer:length")
    assert record["task"] == "lehmer:length"
    assert "<LEHMER_START>" in record["tokens"]

    # The same descents-source permutation can be relabeled with another
    # frozen task, which is how every grid cell receives matched inputs.
    source["task"] = "descents"
    relabeled = _record(source, "cycle", "parity")
    assert relabeled["task"] == "cycle:parity"
    assert relabeled["answer"] == ops.parity((3, 1, 4, 2))


def test_transfer_model_uses_only_the_permutation_vocabulary() -> None:
    from neurips_permutations.passage import PERMUTATION20_VOCABULARY, VOCABULARY

    assert len(PERMUTATION20_VOCABULARY) == 163
    assert len(VOCABULARY) == 188


def test_full_verifier_rejects_a_consistently_wrong_answer(tmp_path: Path) -> None:
    source = {
        "id": 0,
        "task": "descents",
        "inputs": {"primary": [3, 1, 4, 2]},
        "answer": 2,
    }
    record = _record(source, "one_line")
    record["answer"] = 999
    shard = tmp_path / "train-part-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    manifest = {
        "schema_version": "permutation-representation-transfer-32/v1",
        "split": "train",
        "tasks": list(TRAIN_COMBINATIONS),
        "count": 1,
        "task_counts": {"one_line:descents": 1},
        "total_bytes": shard.stat().st_size,
        "shards": [
            {
                "filename": shard.name,
                "byte_size": shard.stat().st_size,
                "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
                "record_count": 1,
                "task_counts": {"one_line:descents": 1},
            }
        ],
    }
    path = tmp_path / "train_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mathematical truth"):
        verify_manifest(path, full=True)


def test_cell_summary_averages_only_across_seeds() -> None:
    rows = []
    for combination in FULL_COMBINATIONS:
        representation, task = combination.split(":", 1)
        for index, seed in enumerate((17, 42, 314159), start=1):
            rows.append(
                {
                    "seed": seed,
                    "representation": representation,
                    "task": task,
                    "combination": combination,
                    "loss": float(index),
                    "token_accuracy": index / 10,
                    "sequence_accuracy": index / 20,
                    "majority_baseline_sequence_accuracy": 0.05,
                    "sequence_accuracy_minus_majority": index / 20 - 0.05,
                }
            )
    summary = summarize_cells(rows)
    assert len(summary) == 32
    assert summary[0]["seed_count"] == 3
    assert summary[0]["loss_mean"] == 2.0
    assert summary[0]["sequence_accuracy_mean"] == pytest.approx(0.1)
    assert summary[0]["sequence_accuracy_minus_majority_mean"] == pytest.approx(0.05)

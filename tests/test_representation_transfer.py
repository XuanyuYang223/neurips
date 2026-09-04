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
    assert record["id"] == 7 * 4 + 2
    assert record["task"] == "lehmer:length"
    assert "<LEHMER_START>" in record["tokens"]

    # The same descents-source permutation can be relabeled with another
    # frozen task, which is how every grid cell receives matched inputs.
    source["task"] = "descents"
    relabeled = _record(source, "cycle", "parity")
    assert relabeled["task"] == "cycle:parity"
    assert relabeled["answer"] == ops.parity((3, 1, 4, 2))


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

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from neurips_permutations.evaluate import (
    EVALUATION_FORMAT_VERSION,
    _load_completed_result,
    build_parser,
    evaluate_all,
    evaluate_records,
    task_homogeneous_batches,
)
from neurips_permutations.passage import TOKEN_TO_ID


def test_task_homogeneous_batches_stream_once_with_both_limits() -> None:
    records = [
        {"task": "a", "tokens": ["x"] * length, "id": index}
        for index, length in enumerate((3, 5, 4, 7, 2))
    ] + [
        {"task": "b", "tokens": ["x"] * length, "id": 10 + index}
        for index, length in enumerate((2, 2, 2))
    ]
    batches = list(
        task_homogeneous_batches(
            records,
            task_names=("a", "b"),
            max_examples=3,
            max_padded_tokens=12,
        )
    )
    flattened = []
    for task, batch in batches:
        assert {record["task"] for record in batch} == {task}
        assert len(batch) <= 3
        assert len(batch) * max(len(record["tokens"]) for record in batch) <= 12
        flattened.extend(record["id"] for record in batch)
    assert sorted(flattened) == sorted(record["id"] for record in records)


def test_evaluator_accepts_a_nested_only_scaling_matrix(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--matrix", "nested"])
    assert args.matrix == "nested"
    with pytest.raises(ValueError, match="nonempty unique"):
        evaluate_all(tmp_path / "missing.toml", tmp_path / "out", matrices=())
    with pytest.raises(ValueError, match="nested or category"):
        evaluate_all(
            tmp_path / "missing.toml",
            tmp_path / "out",
            matrices=("other",),  # type: ignore[arg-type]
        )


class _PerfectAnswerModel(nn.Module):
    def forward(self, input_ids, attention_mask=None):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, len(TOKEN_TO_ID), device=input_ids.device)
        logits[:, 7, TOKEN_TO_ID["01"]] = 10
        logits[:, 8, TOKEN_TO_ID["<EOS>"]] = 10
        return logits


def _record(task: str, identifier: int) -> dict:
    return {
        "id": identifier,
        "task": task,
        "tokens": [
            "<BOS>",
            "<ONE_START>",
            "02",
            ",",
            "01",
            "<ONE_END>",
            "<LENGTH>",
            "=",
            "01",
            "<EOS>",
        ],
    }


def test_evaluate_records_reports_per_task_exact_and_token_accuracy() -> None:
    metrics = evaluate_records(
        _PerfectAnswerModel(),
        [_record("length", 1), _record("peaks", 2)],
        task_names=("length", "peaks"),
        device=torch.device("cpu"),
        max_seq_len=16,
        max_examples=4,
        max_padded_tokens=64,
        amp_enabled=False,
    )
    for metric in metrics.values():
        assert metric["examples"] == 1
        assert metric["tokens"] == 2
        assert metric["token_accuracy"] == 1
        assert metric["sequence_accuracy"] == 1
        assert metric["loss"] > 0


def test_completed_result_resume_requires_exact_identity_and_counts(tmp_path: Path) -> None:
    identity = {
        "format_version": EVALUATION_FORMAT_VERSION,
        "run_id": "run",
        "checkpoint_sha256": "c" * 64,
    }
    payload = {
        **identity,
        "status": "completed",
        "metrics": {
            "length": {"examples": 5},
            "peaks": {"examples": 5},
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_completed_result(
        path,
        identity,
        ("length", "peaks"),
        expected_examples_per_task=5,
    ) == payload

    changed = dict(identity, checkpoint_sha256="d" * 64)
    with pytest.raises(ValueError, match="identity differs"):
        _load_completed_result(
            path,
            changed,
            ("length", "peaks"),
            expected_examples_per_task=5,
        )


def test_completed_result_resume_rejects_missing_task(tmp_path: Path) -> None:
    identity = {"format_version": EVALUATION_FORMAT_VERSION, "run_id": "run"}
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                **identity,
                "status": "completed",
                "metrics": {"length": {"examples": 5}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task grid differs"):
        _load_completed_result(
            path,
            identity,
            ("length", "peaks"),
            expected_examples_per_task=5,
        )

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn

from neurips_permutations.passage import TOKEN_TO_ID, passage_tokens
from neurips_permutations.training import (
    AnswerOnlyCollator,
    StreamingPermutationDataset,
    TokenBudgetBatcher,
    TrainConfig,
    resolve_shards,
    task_names_for_data_source,
    train_run,
)
from neurips_permutations.math_ops import V2_TASK_NAMES, V3_TASK_NAMES
from neurips_permutations import training as training_module


def _record(
    record_id: int,
    task: str,
    *,
    primary: tuple[int, ...] = (3, 1, 4, 2),
    answer: object = 2,
    **operands: object,
) -> dict[str, object]:
    tokens = list(passage_tokens(task, primary, answer, **operands))
    return {
        "schema_version": "permutation-20/v1",
        "id": record_id,
        "task": task,
        "n": len(primary),
        "tokens": tokens,
        "canonical_text": " ".join(tokens),
    }


def _write_shard(path: Path, records: list[dict[str, object]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
    return path


def _make_shards(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        _record(index, "descents" if index % 2 == 0 else "length", answer=index % 4)
        for index in range(12)
    ]
    return (
        _write_shard(tmp_path / "shard-00000.jsonl.gz", records[:6]),
        _write_shard(tmp_path / "shard-00001.jsonl.gz", records[6:]),
    )


def test_streaming_filtering_and_shard_selection(tmp_path: Path) -> None:
    shards = _make_shards(tmp_path)
    dataset = StreamingPermutationDataset(
        shards,
        tasks=("descents",),
        shard_indices=(1,),
        shuffle_buffer_size=1,
    )
    records = list(dataset)
    assert [record["id"] for record in records] == [6, 8, 10]
    assert all(record["task"] == "descents" for record in records)


def test_manifest_filename_resolution(tmp_path: Path) -> None:
    shards = _make_shards(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"shards": [{"filename": shard.name} for shard in shards]}),
        encoding="utf-8",
    )
    assert resolve_shards(manifest) == shards


def test_formal_manifest_selects_schema_aware_validation_tasks(tmp_path: Path) -> None:
    v3_manifest = tmp_path / "v3-manifest.json"
    v3_manifest.write_text(
        json.dumps(
            {
                "schema_version": "permutation-20/v3",
                "tasks": list(V3_TASK_NAMES),
                "shards": [],
            }
        ),
        encoding="utf-8",
    )
    assert task_names_for_data_source(v3_manifest) == V3_TASK_NAMES

    legacy_manifest = tmp_path / "legacy-manifest.json"
    legacy_manifest.write_text(json.dumps({"shards": []}), encoding="utf-8")
    assert task_names_for_data_source(legacy_manifest) == V2_TASK_NAMES

    value = json.loads(v3_manifest.read_text(encoding="utf-8"))
    value["tasks"] = list(V2_TASK_NAMES)
    v3_manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="task registry"):
        task_names_for_data_source(v3_manifest)


def test_v3_manifest_stream_filters_the_new_statistics(tmp_path: Path) -> None:
    shard = _write_shard(
        tmp_path / "part-00000.jsonl.gz",
        [
            _record(0, "peaks", answer=1),
            _record(1, "exceedances", answer=2),
            _record(2, "recoils", answer=1),
            _record(3, "length", answer=3),
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "permutation-20/v3",
                "tasks": list(V3_TASK_NAMES),
                "shards": [{"filename": shard.name}],
            }
        ),
        encoding="utf-8",
    )

    records = list(
        StreamingPermutationDataset(
            manifest,
            tasks=("peaks", "recoils"),
            shuffle_buffer_size=1,
        )
    )
    assert [(record["id"], record["task"]) for record in records] == [
        (0, "peaks"),
        (2, "recoils"),
    ]


def test_bounded_shuffle_is_deterministic_and_epoch_specific(tmp_path: Path) -> None:
    shards = _make_shards(tmp_path)
    first = StreamingPermutationDataset(
        shards, seed=17, epoch=3, shuffle_buffer_size=5
    )
    repeat = StreamingPermutationDataset(
        shards, seed=17, epoch=3, shuffle_buffer_size=5
    )
    next_epoch = StreamingPermutationDataset(
        shards, seed=17, epoch=4, shuffle_buffer_size=5
    )
    first_ids = [record["id"] for record in first]
    assert first_ids == [record["id"] for record in repeat]
    assert first_ids != [record["id"] for record in next_epoch]
    assert sorted(first_ids) == list(range(12))


def test_rank_partitions_are_disjoint_and_complete(tmp_path: Path) -> None:
    shards = _make_shards(tmp_path)
    rank_zero = StreamingPermutationDataset(
        shards, rank=0, world_size=2, shuffle_buffer_size=1
    )
    rank_one = StreamingPermutationDataset(
        shards, rank=1, world_size=2, shuffle_buffer_size=1
    )
    zero_ids = {record["id"] for record in rank_zero}
    one_ids = {record["id"] for record in rank_one}
    assert zero_ids.isdisjoint(one_ids)
    assert zero_ids | one_ids == set(range(12))


def test_token_budget_reduces_batch_size_without_truncating(tmp_path: Path) -> None:
    shards = _make_shards(tmp_path)
    dataset = StreamingPermutationDataset(shards, shuffle_buffer_size=1)
    sequence_length = len(_record(0, "descents")["tokens"])
    batches = list(
        TokenBudgetBatcher(
            dataset,
            max_examples=10,
            max_padded_tokens=sequence_length * 3,
        )
    )
    assert [len(batch) for batch in batches] == [3, 3, 3, 3]
    assert sum(len(batch) for batch in batches) == 12


def test_collator_labels_only_answer_through_eos() -> None:
    first = _record(1, "descents", answer=2)
    second = _record(2, "length", answer=100)
    batch = AnswerOnlyCollator(max_seq_len=64)([first, second])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].dtype == torch.bool
    for row_index, record in enumerate((first, second)):
        tokens = record["tokens"]
        equals = tokens.index("=")
        length = len(tokens)
        assert batch["labels"][row_index, : equals + 1].eq(-100).all()
        expected = torch.tensor(
            [TOKEN_TO_ID[token] for token in tokens[equals + 1 :]], dtype=torch.long
        )
        assert torch.equal(batch["labels"][row_index, equals + 1 : length], expected)
        assert batch["labels"][row_index, length:].eq(-100).all()
        assert tokens[-1] == "<EOS>"


def test_collator_never_truncates_an_answer() -> None:
    record = _record(1, "descents", answer=2)
    with pytest.raises(ValueError, match="never truncated"):
        AnswerOnlyCollator(max_seq_len=8)([record])


def test_training_loss_weights_examples_not_answer_token_counts() -> None:
    # Example zero has one supervised target; example one has four.  Their
    # per-example cross-entropies must contribute equally to the train loss.
    logits = torch.zeros(2, 5, 3)
    labels = torch.full((2, 5), -100, dtype=torch.long)
    labels[0, 1] = 0
    labels[1, 1:] = 1
    logits[0, 0, 0] = 3.0
    logits[1, :4, 1] = 1.0
    loss = training_module._causal_loss(logits, labels)
    first = torch.nn.functional.cross_entropy(logits[0, 0:1], labels[0, 1:2])
    second = torch.nn.functional.cross_entropy(logits[1, :4], labels[1, 1:])
    torch.testing.assert_close(loss, (first + second) / 2)


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(TOKEN_TO_ID), 12)
        self.output = nn.Linear(12, len(TOKEN_TO_ID))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del attention_mask
        return {"logits": self.output(self.embedding(input_ids))}


def _tiny_factory(config: TrainConfig) -> nn.Module:
    del config
    return _TinyLM()


def _training_config(shard: Path, output_dir: Path, *, max_steps: int) -> TrainConfig:
    return TrainConfig(
        output_dir=str(output_dir),
        train_shards=(str(shard),),
        architecture="mlp",
        d_model=8,
        num_layers=1,
        num_heads=2,
        tasks=("descents", "length"),
        validation_tasks=("descents", "length"),
        seed=1234,
        max_steps=max_steps,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_len=64,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_steps=0,
        min_lr_ratio=1.0,
        shuffle_buffer_size=4,
        num_workers=0,
        checkpoint_every=1,
        validate_every=1,
        validation_batches_per_task=1,
        device="cpu",
        amp=False,
    )


def test_one_cpu_train_step_checkpoints_and_validates(tmp_path: Path) -> None:
    records = [
        _record(index, "descents" if index % 2 == 0 else "length", answer=index % 4)
        for index in range(8)
    ]
    shard = _write_shard(tmp_path / "train.jsonl.gz", records)
    output = tmp_path / "run"
    summary = train_run(
        _training_config(shard, output, max_steps=1)
    )
    assert summary["status"] == "completed"
    assert summary["global_step"] == 1
    assert set(summary["validation"]) == {"descents", "length"}
    assert (output / "checkpoint.pt").is_file()
    assert (output / "completed.json").is_file()
    marker = json.loads((output / "completed.json").read_text())
    assert len(marker["checkpoint_sha256"]) == 64
    assert len(marker["config_sha256"]) == 64
    assert marker["architecture"] == "mlp"
    assert marker["tasks"] == ["descents", "length"]
    assert set(marker["task_accounting"]) == {"descents", "length"}
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "scheduler", "rng", "config", "state"} <= set(
        checkpoint
    )


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    records = [
        _record(index, "descents" if index % 2 == 0 else "length", answer=index % 4)
        for index in range(10)
    ]
    shard = _write_shard(tmp_path / "train.jsonl.gz", records)

    full_dir = tmp_path / "full"
    full_config = replace(
        _training_config(shard, full_dir, max_steps=2), validate_every=0
    )
    train_run(full_config, model_factory=_tiny_factory)

    resumed_dir = tmp_path / "resumed"
    staged_config = replace(
        _training_config(shard, resumed_dir, max_steps=2), validate_every=0
    )
    first = train_run(
        staged_config, model_factory=_tiny_factory, stop_after_steps=1
    )
    assert first["status"] == "stopped"
    assert first["global_step"] == 1
    assert not (resumed_dir / "completed.json").exists()
    resumed = train_run(
        replace(staged_config, resume=str(resumed_dir / "checkpoint.pt")),
        model_factory=_tiny_factory,
    )
    assert resumed["status"] == "completed"
    assert resumed["global_step"] == 2

    full_checkpoint = torch.load(
        full_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        resumed_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert full_checkpoint["state"] == resumed_checkpoint["state"]
    for name, tensor in full_checkpoint["model"].items():
        assert torch.equal(tensor, resumed_checkpoint["model"][name]), name


def test_restore_rng_normalizes_cuda_states_to_cpu_byte_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[torch.Tensor] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def capture(values: list[torch.Tensor]) -> None:
        captured.extend(values)

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", capture)
    training_module._restore_rng(
        {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": [[1, 2, 3], torch.tensor([4, 5], dtype=torch.int64)],
        }
    )

    assert [value.tolist() for value in captured] == [[1, 2, 3], [4, 5]]
    assert all(value.device.type == "cpu" for value in captured)
    assert all(value.dtype == torch.uint8 for value in captured)


def test_resume_rejects_changed_training_config(tmp_path: Path) -> None:
    records = [_record(index, "descents", answer=index % 2) for index in range(4)]
    shard = _write_shard(tmp_path / "train.jsonl.gz", records)
    output = tmp_path / "run"
    config = replace(
        _training_config(shard, output, max_steps=2), validate_every=0
    )
    train_run(config, model_factory=_tiny_factory, stop_after_steps=1)
    changed = replace(
        config,
        learning_rate=config.learning_rate * 2,
        resume=str(output / "checkpoint.pt"),
    )
    with pytest.raises(ValueError, match="differs"):
        train_run(changed, model_factory=_tiny_factory)


def test_resume_rejects_changed_manifest_contents(tmp_path: Path) -> None:
    records = [_record(index, "descents", answer=index % 2) for index in range(4)]
    shard = _write_shard(tmp_path / "train.jsonl.gz", records)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"shards": [{"filename": shard.name}]}), encoding="utf-8"
    )
    output = tmp_path / "run"
    config = replace(
        _training_config(shard, output, max_steps=2),
        train_shards=(),
        manifest=str(manifest),
        validate_every=0,
    )
    train_run(config, model_factory=_tiny_factory, stop_after_steps=1)
    manifest.write_text(
        json.dumps({"shards": [{"filename": shard.name}], "changed": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprints"):
        train_run(
            replace(config, resume=str(output / "checkpoint.pt")),
            model_factory=_tiny_factory,
        )

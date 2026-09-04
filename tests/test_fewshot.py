from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import torch

from neurips_permutations.fewshot import (
    FORMAT_VERSION,
    SHOT_CURVE_FORMAT_VERSION,
    SHOT_CURVE_SUPPORT_FORMAT_VERSION,
    SUPPORT_FORMAT_VERSION,
    FewshotSpec,
    _completion_valid,
    _run_identity,
    _select_support_ids,
    _sha256,
    _train_one,
    build_plan,
    load_support_artifact,
)
import neurips_permutations.fewshot as fewshot_module
from neurips_permutations.training import TrainConfig, _default_model_factory


def _spec(tmp_path: Path, **changes: object) -> FewshotSpec:
    placeholder = tmp_path / "placeholder.json"
    placeholder.write_text("{}\n", encoding="utf-8")
    spec = FewshotSpec(
        repository=tmp_path,
        config_path=tmp_path / "fewshot.toml",
        config_sha256="f" * 64,
        base_config_path=tmp_path / "base.toml",
        base_config_sha256="b" * 64,
        train_manifest=placeholder,
        train_manifest_sha256="1" * 64,
        validation_manifest=placeholder,
        validation_manifest_sha256="2" * 64,
        test_manifest=placeholder,
        test_manifest_sha256="3" * 64,
        support_artifact=tmp_path / "support.json",
        output_dir=tmp_path / "runs",
        evaluation_dir=tmp_path / "evaluation",
        holdout_tasks=("to_reduced_word", "compose", "parity", "to_lehmer"),
        model_seeds=(17, 42, 314159),
        shots=20,
        max_steps=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        max_tokens_per_batch=4096,
        learning_rate=1e-5,
        random_init_learning_rate=3e-4,
        min_learning_rate_ratio=0.1,
        weight_decay=0.01,
        warmup_steps=1,
        max_grad_norm=1.0,
        bf16=True,
        expected_validation_examples=5,
        expected_test_examples=5,
        expected_runs=144,
    )
    return replace(spec, **changes)


def _audited_runs(tmp_path: Path) -> list[dict[str, object]]:
    runs = []
    for architecture in ("transformer", "mlp"):
        for task_count in (1, 2, 4, 8, 16):
            for seed in (17, 42, 314159):
                run_id = f"{architecture}-tasks{task_count:02d}-seed{seed}"
                runs.append(
                    {
                        "run_id": run_id,
                        "architecture": architecture,
                        "task_count": task_count,
                        "seed": seed,
                        "status": "passed",
                        "checkpoint_path": str(tmp_path / run_id / "checkpoint.pt"),
                        "checkpoint_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
                    }
                )
    return runs


def test_plan_contains_paired_pretrained_and_random_controls(tmp_path: Path) -> None:
    plan = build_plan(_spec(tmp_path), _audited_runs(tmp_path))
    assert len(plan) == len({run["run_id"] for run in plan}) == 144
    assert sum(run["initialization"] == "pretrained" for run in plan) == 120
    assert sum(run["initialization"] == "random" for run in plan) == 24
    assert {
        (run["architecture"], run["base_trained_task_count"])
        for run in plan
        if run["initialization"] == "pretrained"
    } == {
        (architecture, count)
        for architecture in ("transformer", "mlp")
        for count in (1, 2, 4, 8, 16)
    }
    assert all(
        run["source_checkpoint_sha256"] == "random_init"
        and len(run["model_config_checkpoint_sha256"]) == 64
        for run in plan
        if run["initialization"] == "random"
    )


def _record(record_id: int, task: str = "parity") -> dict[str, object]:
    return {
        "id": record_id,
        "task": task,
        "tokens": [
            "<BOS>",
            "<SIZE>",
            "02",
            "<ONE_START>",
            "01",
            ",",
            "02",
            "<ONE_END>",
            "<PARITY>",
            "=",
            "00",
            "<EOS>",
        ],
    }


def test_support_artifact_rejects_duplicate_or_wrong_task_records(tmp_path: Path) -> None:
    spec = _spec(tmp_path, holdout_tasks=("parity",), model_seeds=(17,))
    records = [_record(index) for index in range(20)]
    artifact = {
        "format_version": SUPPORT_FORMAT_VERSION,
        "fewshot_config_sha256": spec.config_sha256,
        "train_manifest_sha256": spec.train_manifest_sha256,
        "shots": 20,
        "tasks": ["parity"],
        "seeds": [17],
        "set_count": 1,
        "record_count": 20,
        "sets": [
            {
                "key": "parity:seed17",
                "task": "parity",
                "seed": 17,
                "record_ids": list(range(20)),
                "records": records,
            }
        ],
    }
    spec.support_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    assert load_support_artifact(spec)["record_count"] == 20

    artifact["sets"][0]["records"][1]["id"] = 0
    spec.support_artifact.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid records or IDs"):
        load_support_artifact(spec)


def test_shot_curve_support_is_strictly_nested_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = ("to_reduced_word", "compose", "parity", "to_lehmer")
    seeds = (17, 42, 314159)
    anchor: dict[tuple[str, int], list[dict[str, int]]] = {}
    for seed_index, seed in enumerate(seeds):
        for task_index, task in enumerate(tasks):
            start = 1_000 * (seed_index * len(tasks) + task_index)
            anchor[(task, seed)] = [
                {"id": task_index + len(tasks) * (start + offset)}
                for offset in range(20)
            ]
    monkeypatch.setattr(fewshot_module, "_anchor_support_lookup", lambda spec: anchor)
    common = {
        "protocol_version": SHOT_CURVE_FORMAT_VERSION,
        "support_format_version": SHOT_CURVE_SUPPORT_FORMAT_VERSION,
        "anchor_support_artifact": tmp_path / "anchor.json",
        "anchor_support_sha256": "a" * 64,
    }
    spec5 = _spec(tmp_path, shots=5, **common)
    spec100 = _spec(tmp_path, shots=100, **common)
    counts = {task: 100_000 for task in tasks}
    selected5 = _select_support_ids(spec5, task_names=tasks, task_counts=counts)
    selected100 = _select_support_ids(spec100, task_names=tasks, task_counts=counts)
    assert selected100 == _select_support_ids(
        spec100, task_names=tasks, task_counts=counts
    )
    for seed in seeds:
        for task in tasks:
            key = f"{task}:seed{seed}"
            anchor_ids = [record["id"] for record in anchor[(task, seed)]]
            assert selected5[key] == anchor_ids[:5]
            assert selected100[key][:20] == anchor_ids
            assert len(set(selected100[key])) == 100
    assert len({item for values in selected100.values() for item in values}) == 1200


def test_shot_curve_run_identity_uses_protocol_from_spec(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        shots=5,
        protocol_version=SHOT_CURVE_FORMAT_VERSION,
        support_format_version=SHOT_CURVE_SUPPORT_FORMAT_VERSION,
    )
    run = build_plan(spec, _audited_runs(tmp_path))[0]
    identity = _run_identity(
        spec,
        run,
        support_sha256="a" * 64,
        support_ids=range(5),
        implementation_commit="c" * 40,
    )
    assert identity["format_version"] == SHOT_CURVE_FORMAT_VERSION
    assert identity["shots"] == 5


def test_one_adaptation_warm_starts_trains_and_validates(tmp_path: Path) -> None:
    records = [_record(index) for index in range(20)]
    validation_shard = tmp_path / "part-00000.jsonl.gz"
    with gzip.open(validation_shard, "wt", encoding="utf-8") as handle:
        for record in records[:5]:
            handle.write(json.dumps(record) + "\n")
    validation_manifest = tmp_path / "validation_manifest.json"
    validation_manifest.write_text(
        json.dumps({"shards": [{"filename": validation_shard.name}]}),
        encoding="utf-8",
    )
    spec = _spec(tmp_path, validation_manifest=validation_manifest)
    train_config = TrainConfig(
        output_dir=str(tmp_path / "base"),
        train_shards=(str(validation_shard),),
        architecture="transformer",
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        mlp_ratio=2.0,
        max_seq_len=32,
        amp=False,
    )
    source_model = _default_model_factory(train_config)
    source_checkpoint = tmp_path / "source.pt"
    torch.save(
        {"config": asdict(train_config), "model": source_model.state_dict()},
        source_checkpoint,
    )
    source_sha256 = _sha256(source_checkpoint)
    run = {
        "run_id": "pretrained-transformer-tasks01-seed17-parity",
        "initialization": "pretrained",
        "architecture": "transformer",
        "base_run_id": "transformer-tasks01-seed17",
        "base_trained_task_count": 1,
        "seed": 17,
        "task": "parity",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "model_config_checkpoint_sha256": source_sha256,
    }
    marker = _train_one(
        spec,
        run,
        records,
        support_sha256="a" * 64,
        implementation_commit="c" * 40,
        device=torch.device("cpu"),
    )
    checkpoint_path = spec.output_dir / run["run_id"] / "checkpoint.pt"
    marker_path = checkpoint_path.with_name("completed.json")
    assert marker["status"] == "completed"
    assert marker["train_examples"] == spec.max_steps * spec.batch_size == 8
    assert marker["validation"]["examples"] == 5
    identity = {
        key: value
        for key, value in marker.items()
        if key
        not in {
            "status",
            "checkpoint",
            "checkpoint_sha256",
            "validation",
            "train_mean_example_loss",
            "train_examples",
            "elapsed_seconds",
        }
    }
    assert _completion_valid(marker_path, identity, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["format_version"] == FORMAT_VERSION
    assert "optimizer" not in checkpoint
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values())

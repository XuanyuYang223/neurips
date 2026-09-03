"""Balanced Henry-style 20-shot adaptation for the Property32 model matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .evaluate import evaluate_records
from .fewshot import (
    _atomic_json,
    _completion_valid,
    _git_commit,
    _locked,
    _portable_path,
    _run_identity,
    _sha256,
    _stable_seed,
    _train_one,
)
from .property_experiments import build_property_matrix, matrix_summary
from .property_replicates import PROPERTY_FAMILIES
from .training import StreamingPermutationDataset, TrainConfig, _default_model_factory, resolve_shards
from .verify import verify_manifest


DEFAULT_CONFIG = Path("configs/property32_fewshot.toml")
FORMAT_VERSION = "property32-fewshot/v1"
SUPPORT_FORMAT_VERSION = "property32-fewshot-support/v1"
TEST_FORMAT_VERSION = "property32-fewshot-test/v1"
ZERO_SHOT_FORMAT_VERSION = "property32-fewshot-zero-shot/v1"
REPLICATE_IDS = ("r0", "r1", "r2")
TASK_COUNTS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PropertyFewshotSpec:
    repository: Path
    config_path: Path
    config_sha256: str
    train_manifest: Path
    train_manifest_sha256: str
    validation_manifest: Path
    validation_manifest_sha256: str
    test_manifest: Path
    test_manifest_sha256: str
    support_artifact: Path
    output_dir: Path
    evaluation_dir: Path
    shots: int
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    max_tokens_per_batch: int
    learning_rate: float
    random_init_learning_rate: float
    min_learning_rate_ratio: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    bf16: bool
    expected_validation_examples: int
    expected_test_examples: int
    expected_runs: int
    replicate_configs: Mapping[str, Path]
    replicate_config_sha256: Mapping[str, str]
    replicate_seeds: Mapping[str, int]
    target_sets: Mapping[tuple[str, str], tuple[str, ...]]
    test_shard_position: int
    test_source_shard_index: int
    test_source_shard_sha256: str


def _family(task: str) -> str:
    names = ("local", "positional", "cycle", "global_run")
    for name, tasks in zip(names, PROPERTY_FAMILIES, strict=True):
        if task in tasks:
            return name
    raise ValueError(f"property has no family: {task}")


def load_spec(config_path: Path = DEFAULT_CONFIG) -> PropertyFewshotSpec:
    config_path = config_path.resolve()
    repository = config_path.parent.parent
    payload = config_path.read_bytes()
    value = tomllib.loads(payload.decode("utf-8"))
    if value.get("protocol_version") != FORMAT_VERSION:
        raise ValueError("unsupported Property32 few-shot protocol")

    def path(raw: str) -> Path:
        return (repository / raw).resolve()

    replicate_configs: dict[str, Path] = {}
    replicate_hashes: dict[str, str] = {}
    replicate_seeds: dict[str, int] = {}
    targets: dict[tuple[str, str], tuple[str, ...]] = {}
    declared = value.get("base_replicates")
    if not isinstance(declared, Mapping) or tuple(declared) != REPLICATE_IDS:
        raise ValueError("Property32 few-shot requires frozen r0/r1/r2 replicates")
    for replicate_id in REPLICATE_IDS:
        item = declared[replicate_id]
        replicate_configs[replicate_id] = path(str(item["config"]))
        replicate_hashes[replicate_id] = str(item["config_sha256"])
        replicate_seeds[replicate_id] = int(item["model_seed"])
        for pool in ("a", "b"):
            selected = tuple(item[f"pool_{pool}_model_targets"])
            if len(selected) != 4 or len(set(selected)) != 4:
                raise ValueError("each pool direction requires four unique targets")
            if {_family(task) for task in selected} != {
                "local", "positional", "cycle", "global_run"
            }:
                raise ValueError("each target set must contain one property per family")
            targets[(replicate_id, pool)] = selected

    fine = value["fine_tuning"]
    evaluation = value["evaluation"]
    spec = PropertyFewshotSpec(
        repository=repository,
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        train_manifest=path(str(value["train_manifest"])),
        train_manifest_sha256=str(value["train_manifest_sha256"]),
        validation_manifest=path(str(value["validation_manifest"])),
        validation_manifest_sha256=str(value["validation_manifest_sha256"]),
        test_manifest=path(str(value["test_manifest"])),
        test_manifest_sha256=str(value["test_manifest_sha256"]),
        support_artifact=path(str(value["support_artifact"])),
        output_dir=path(str(value["output_dir"])),
        evaluation_dir=path(str(value["evaluation_dir"])),
        shots=int(value["shots"]),
        max_steps=int(fine["max_steps"]),
        batch_size=int(fine["micro_batch_size"]),
        gradient_accumulation_steps=int(fine["gradient_accumulation_steps"]),
        max_tokens_per_batch=int(fine["max_tokens_per_batch"]),
        learning_rate=float(fine["learning_rate"]),
        random_init_learning_rate=float(fine["random_init_learning_rate"]),
        min_learning_rate_ratio=float(fine["min_learning_rate_ratio"]),
        weight_decay=float(fine["weight_decay"]),
        warmup_steps=int(fine["warmup_steps"]),
        max_grad_norm=float(fine["gradient_clip_norm"]),
        bf16=str(fine["precision"]).lower() in {"bf16", "bfloat16"},
        expected_validation_examples=int(evaluation["validation_examples_per_task"]),
        expected_test_examples=int(evaluation["test_examples_per_task"]),
        expected_runs=int(value["matrix"]["total_runs"]),
        replicate_configs=replicate_configs,
        replicate_config_sha256=replicate_hashes,
        replicate_seeds=replicate_seeds,
        target_sets=targets,
        test_shard_position=int(value["test_shard_position"]),
        test_source_shard_index=int(value["test_source_shard_index"]),
        test_source_shard_sha256=str(value["test_source_shard_sha256"]),
    )
    if spec.shots != 20 or spec.expected_runs != 144:
        raise ValueError("Property32 pilot is frozen at 20 shots and 144 runs")
    if tuple(spec.replicate_seeds.values()) != (17, 42, 101):
        raise ValueError("Property32 few-shot model seeds differ")
    if (
        spec.max_steps != 200
        or spec.batch_size != 4
        or spec.gradient_accumulation_steps != 1
        or spec.warmup_steps != 20
    ):
        raise ValueError("Property32 few-shot update schedule differs")
    if spec.expected_validation_examples != 5_000 or spec.expected_test_examples != 2_500:
        raise ValueError("Property32 few-shot evaluation counts differ")
    if not 0 < spec.learning_rate < spec.random_init_learning_rate:
        raise ValueError("warm-start learning rate must be lower than random-init rate")
    for manifest, expected in (
        (spec.train_manifest, spec.train_manifest_sha256),
        (spec.validation_manifest, spec.validation_manifest_sha256),
        (spec.test_manifest, spec.test_manifest_sha256),
    ):
        if _sha256(manifest) != expected:
            raise ValueError(f"manifest differs from frozen config: {manifest}")
    for replicate_id, path_value in spec.replicate_configs.items():
        if _sha256(path_value) != spec.replicate_config_sha256[replicate_id]:
            raise ValueError(f"base replicate config differs: {replicate_id}")
        raw = tomllib.loads(path_value.read_text(encoding="utf-8"))
        for pool in ("a", "b"):
            opposite = "b" if pool == "a" else "a"
            expected = tuple(raw[f"pool_{opposite}"][:4])
            if spec.target_sets[(replicate_id, pool)] != expected:
                raise ValueError("target set is not the first balanced opposite-pool block")
    test_manifest = json.loads(spec.test_manifest.read_text(encoding="utf-8"))
    entry = test_manifest["shards"][spec.test_shard_position]
    if (
        int(entry["index"]) != spec.test_source_shard_index
        or entry["sha256"] != spec.test_source_shard_sha256
        or int(entry["record_count"]) != 80_000
        or set(entry["task_counts"].values()) != {spec.expected_test_examples}
    ):
        raise ValueError("frozen source shard 199 metadata differs")
    return spec


def _support_specs(spec: PropertyFewshotSpec) -> tuple[dict[str, Any], ...]:
    result = []
    for replicate_id in REPLICATE_IDS:
        seed = spec.replicate_seeds[replicate_id]
        for pool in ("a", "b"):
            for task in spec.target_sets[(replicate_id, pool)]:
                result.append(
                    {
                        "key": f"{replicate_id}:pool{pool}:{task}",
                        "replicate_id": replicate_id,
                        "model_pool": pool,
                        "seed": seed,
                        "task": task,
                        "target_family": _family(task),
                    }
                )
    if len(result) != 24 or len({item["key"] for item in result}) != 24:
        raise ValueError("support design must contain 24 unique target cells")
    return tuple(result)


def build_support_artifact(
    spec: PropertyFewshotSpec, *, overwrite: bool = False
) -> dict[str, Any]:
    if spec.support_artifact.is_file() and not overwrite:
        return load_support_artifact(spec)
    manifest = json.loads(spec.train_manifest.read_text(encoding="utf-8"))
    if manifest.get("count") != 15_680_000 or len(manifest.get("shards", ())) != 196:
        raise ValueError("support examples must come from the 15.68M training split")
    task_names = tuple(manifest["tasks"])
    used_ids: set[int] = set()
    selected_ids: dict[str, list[int]] = {}
    desired: dict[int, str] = {}
    for item in _support_specs(spec):
        task = str(item["task"])
        available = int(manifest["task_counts"][task])
        task_index = task_names.index(task)
        rng = random.Random(
            _stable_seed(
                SUPPORT_FORMAT_VERSION,
                item["replicate_id"],
                item["model_pool"],
                task,
                item["seed"],
            )
        )
        ids: set[int] = set()
        while len(ids) < spec.shots:
            record_id = task_index + len(task_names) * rng.randrange(available)
            if record_id not in used_ids:
                ids.add(record_id)
                used_ids.add(record_id)
        ordered = sorted(ids)
        selected_ids[str(item["key"])] = ordered
        desired.update({record_id: str(item["key"]) for record_id in ordered})

    found: dict[int, dict[str, Any]] = {}
    for shard in resolve_shards(spec.train_manifest):
        wanted = desired.keys() - found.keys()
        if not wanted:
            break
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_id = int(record["id"])
                if record_id in desired:
                    found[record_id] = record
    if set(found) != set(desired):
        raise ValueError("not all Property32 support records were found")
    sets = []
    for item in _support_specs(spec):
        ids = selected_ids[str(item["key"])]
        records = [found[record_id] for record_id in ids]
        if any(record["task"] != item["task"] for record in records):
            raise ValueError("support records contain the wrong property")
        sets.append({**item, "record_ids": ids, "records": records})
    artifact = {
        "format_version": SUPPORT_FORMAT_VERSION,
        "fewshot_config_sha256": spec.config_sha256,
        "train_manifest_sha256": spec.train_manifest_sha256,
        "selection_method": "seeded_uniform_occurrence_without_replacement/v1",
        "shots": spec.shots,
        "set_count": 24,
        "record_count": 24 * spec.shots,
        "sets": sets,
    }
    _atomic_json(artifact, spec.support_artifact)
    return artifact


def load_support_artifact(spec: PropertyFewshotSpec) -> dict[str, Any]:
    value = json.loads(spec.support_artifact.read_text(encoding="utf-8"))
    expected = {
        "format_version": SUPPORT_FORMAT_VERSION,
        "fewshot_config_sha256": spec.config_sha256,
        "train_manifest_sha256": spec.train_manifest_sha256,
        "shots": 20,
        "set_count": 24,
        "record_count": 480,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Property32 support artifact identity differs")
    expected_cells = {item["key"] for item in _support_specs(spec)}
    seen_ids: set[int] = set()
    seen_cells: set[str] = set()
    for item in value.get("sets", ()):
        key = str(item.get("key"))
        records = item.get("records")
        ids = item.get("record_ids")
        if key not in expected_cells or key in seen_cells or not isinstance(records, list):
            raise ValueError("Property32 support cell differs")
        if len(records) != spec.shots or ids != [record.get("id") for record in records]:
            raise ValueError("Property32 support record geometry differs")
        record_ids = {int(record["id"]) for record in records}
        if len(record_ids) != spec.shots or seen_ids & record_ids:
            raise ValueError("Property32 support records are not globally unique")
        if any(record.get("task") != item.get("task") for record in records):
            raise ValueError("Property32 support task differs")
        seen_ids.update(record_ids)
        seen_cells.add(key)
    if seen_cells != expected_cells:
        raise ValueError("Property32 support artifact is incomplete")
    return value


def _support_lookup(artifact: Mapping[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    return {
        (str(item["replicate_id"]), str(item["model_pool"]), str(item["task"])): list(item["records"])
        for item in artifact["sets"]
    }


def _base_runs(spec: PropertyFewshotSpec, *, strict: bool = True) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for replicate_id in REPLICATE_IDS:
        config_path = spec.replicate_configs[replicate_id]
        status = matrix_summary(config_path)
        if status["complete_count"] != 10:
            raise ValueError(f"base replicate is incomplete: {replicate_id}")
        for run in build_property_matrix(config_path):
            marker_path = Path(run.output_dir) / "completed.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            checkpoint = Path(str(marker["checkpoint"]))
            if not checkpoint.is_file():
                checkpoint = marker_path.parent / checkpoint.name
            if _sha256(checkpoint) != marker.get("checkpoint_sha256"):
                raise ValueError(f"base checkpoint hash differs: {run.run_id}")
            if strict:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                config = TrainConfig.from_value(payload["config"])
                if (
                    config.architecture != "transformer"
                    or tuple(config.tasks) != run.tasks
                    or config.seed != run.seed
                ):
                    raise ValueError(f"base checkpoint config differs: {run.run_id}")
                model = _default_model_factory(config)
                model.load_state_dict(payload["model"], strict=True)
                if not all(torch.isfinite(tensor).all().item() for tensor in payload["model"].values()):
                    raise ValueError(f"base checkpoint is non-finite: {run.run_id}")
                del model, payload
            result.append(
                {
                    "status": "passed",
                    "replicate_id": replicate_id,
                    "model_pool": run.pool,
                    "run_id": run.run_id,
                    "architecture": run.architecture,
                    "task_count": run.task_count,
                    "tasks": list(run.tasks),
                    "seed": run.seed,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": marker["checkpoint_sha256"],
                }
            )
    if len(result) != 30 or len({run["run_id"] for run in result}) != 30:
        raise ValueError("base Property32 matrix must contain 30 unique models")
    return result


def build_plan(spec: PropertyFewshotSpec, base_runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(base_runs) != 30:
        raise ValueError("Property32 few-shot plan requires 30 base models")
    plan: list[dict[str, Any]] = []
    ordered = sorted(
        base_runs,
        key=lambda run: (
            REPLICATE_IDS.index(str(run["replicate_id"])),
            str(run["model_pool"]),
            int(run["task_count"]),
        ),
    )
    for run in ordered:
        if run.get("status") != "passed":
            raise ValueError("Property32 base run did not pass authentication")
        targets = spec.target_sets[(str(run["replicate_id"]), str(run["model_pool"]))]
        if set(targets) & set(run["tasks"]):
            raise ValueError("few-shot target occurred in base-model training")
        for task in targets:
            plan.append(
                {
                    "run_id": f"pretrained-{run['run_id']}-{task}",
                    "initialization": "pretrained",
                    "architecture": "transformer",
                    "base_run_id": run["run_id"],
                    "base_trained_task_count": int(run["task_count"]),
                    "seed": int(run["seed"]),
                    "task": task,
                    "source_checkpoint": run["checkpoint_path"],
                    "source_checkpoint_sha256": run["checkpoint_sha256"],
                    "model_config_checkpoint_sha256": run["checkpoint_sha256"],
                    "replicate_id": run["replicate_id"],
                    "model_pool": run["model_pool"],
                    "target_family": _family(task),
                }
            )
    for replicate_id in REPLICATE_IDS:
        for pool in ("a", "b"):
            template = next(
                run
                for run in ordered
                if run["replicate_id"] == replicate_id
                and run["model_pool"] == pool
                and int(run["task_count"]) == 1
            )
            for task in spec.target_sets[(replicate_id, pool)]:
                plan.append(
                    {
                        "run_id": f"random-{replicate_id}-pool{pool}-seed{template['seed']}-{task}",
                        "initialization": "random",
                        "architecture": "transformer",
                        "base_run_id": "",
                        "base_trained_task_count": 0,
                        "seed": int(template["seed"]),
                        "task": task,
                        "source_checkpoint": template["checkpoint_path"],
                        "source_checkpoint_sha256": "random_init",
                        "model_config_checkpoint_sha256": template["checkpoint_sha256"],
                        "replicate_id": replicate_id,
                        "model_pool": pool,
                        "target_family": _family(task),
                    }
                )
    if len(plan) != 144 or len({run["run_id"] for run in plan}) != 144:
        raise ValueError("Property32 few-shot plan must contain 144 unique runs")
    return plan


def run_all(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    spec = load_spec(config_path)
    implementation_commit = _git_commit(spec.repository)
    artifact = load_support_artifact(spec)
    support_sha256 = _sha256(spec.support_artifact)
    support = _support_lookup(artifact)
    plan = build_plan(spec, _base_runs(spec))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    lock_path = spec.output_dir / ".controller.lock"
    descriptor = _locked(lock_path)
    os.close(descriptor)
    markers = []
    try:
        for index, run in enumerate(plan, start=1):
            print(f"property few-shot {index}/{len(plan)}: {run['run_id']}", flush=True)
            markers.append(
                _train_one(
                    spec,
                    run,
                    support[(str(run["replicate_id"]), str(run["model_pool"]), str(run["task"]))],
                    support_sha256=support_sha256,
                    implementation_commit=implementation_commit,
                    device=device,
                    format_version=FORMAT_VERSION,
                )
            )
        manifest = {
            "format_version": FORMAT_VERSION,
            "status": "completed",
            "implementation_commit": implementation_commit,
            "fewshot_config_sha256": spec.config_sha256,
            "support_artifact_sha256": support_sha256,
            "run_count": len(markers),
            "validation_model_examples": sum(int(marker["validation"]["examples"]) for marker in markers),
            "runs": [
                {"run_id": marker["run_id"], "checkpoint_sha256": marker["checkpoint_sha256"]}
                for marker in markers
            ],
        }
        _atomic_json(manifest, spec.output_dir / "manifest.json")
        return manifest
    finally:
        lock_path.unlink(missing_ok=True)


def audit_all(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    spec = load_spec(config_path)
    artifact = load_support_artifact(spec)
    support_sha256 = _sha256(spec.support_artifact)
    support = _support_lookup(artifact)
    plan = build_plan(spec, _base_runs(spec))
    passed: list[dict[str, Any]] = []
    incomplete: list[str] = []
    failed: list[dict[str, Any]] = []
    for run in plan:
        run_dir = spec.output_dir / str(run["run_id"])
        marker_path = run_dir / "completed.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        if not marker_path.is_file() or not checkpoint_path.is_file():
            incomplete.append(str(run["run_id"]))
            continue
        issues: list[str] = []
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            implementation_commit = marker.get("implementation_commit")
            if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
                issues.append("invalid implementation commit")
            records = support[(str(run["replicate_id"]), str(run["model_pool"]), str(run["task"]))]
            identity = _run_identity(
                spec,
                run,
                support_sha256=support_sha256,
                support_ids=[int(record["id"]) for record in records],
                implementation_commit=str(implementation_commit),
                format_version=FORMAT_VERSION,
            )
            if not _completion_valid(marker_path, identity, checkpoint_path):
                issues.append("completion identity or checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if any(checkpoint.get(key) != expected for key, expected in identity.items()):
                issues.append("checkpoint identity mismatch")
            config = TrainConfig.from_value(checkpoint["base_train_config"])
            if config.architecture != "transformer":
                issues.append("adapted model is not a Transformer")
            model = _default_model_factory(config)
            model.load_state_dict(checkpoint["model"], strict=True)
            if not all(torch.isfinite(tensor).all().item() for tensor in checkpoint["model"].values()):
                issues.append("non-finite model tensor")
            validation = marker.get("validation", {})
            if checkpoint.get("validation") != validation:
                issues.append("marker/checkpoint validation mismatch")
            if validation.get("examples") != spec.expected_validation_examples:
                issues.append("validation example count mismatch")
            for metric in ("loss", "token_accuracy", "sequence_accuracy"):
                metric_value = validation.get(metric)
                if not isinstance(metric_value, (int, float)) or not math.isfinite(float(metric_value)):
                    issues.append(f"invalid validation {metric}")
                elif metric != "loss" and not 0 <= float(metric_value) <= 1:
                    issues.append(f"out-of-range validation {metric}")
            if marker.get("train_examples") != spec.max_steps * spec.batch_size:
                issues.append("training example count mismatch")
            expected_lr = (
                spec.learning_rate if run["initialization"] == "pretrained" else spec.random_init_learning_rate
            ) * spec.min_learning_rate_ratio
            if not math.isclose(float(checkpoint.get("final_learning_rate", -1)), expected_lr, rel_tol=1e-9):
                issues.append("final learning rate mismatch")
            del model, checkpoint
        except Exception as error:
            issues.append(f"{type(error).__name__}: {error}")
        if issues:
            failed.append({"run_id": run["run_id"], "issues": issues})
        else:
            passed.append({**run, "checkpoint": str(checkpoint_path), "checkpoint_sha256": marker["checkpoint_sha256"]})
    partials = sorted(str(path) for path in spec.output_dir.rglob("*.tmp")) if spec.output_dir.exists() else []
    return {
        "format_version": FORMAT_VERSION,
        "status": "passed" if len(passed) == 144 and not failed and not partials else "failed" if failed else "incomplete",
        "ok": len(passed) == 144 and not failed and not partials,
        "expected_run_count": 144,
        "passed_count": len(passed),
        "incomplete_count": len(incomplete),
        "failed_count": len(failed),
        "partial_artifacts": partials,
        "passed": passed,
        "incomplete": incomplete,
        "failed": failed,
    }


def _metric_identity(
    *,
    format_version: str,
    run_id: str,
    checkpoint_sha256: str,
    spec: PropertyFewshotSpec,
    evaluator_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": format_version,
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "fewshot_config_sha256": spec.config_sha256,
        "test_manifest_sha256": spec.test_manifest_sha256,
        "test_shard_position": spec.test_shard_position,
        "test_source_shard_index": spec.test_source_shard_index,
        "test_source_shard_sha256": spec.test_source_shard_sha256,
        "evaluator_commit": evaluator_commit,
    }


def _load_metric_result(path: Path, identity: Mapping[str, Any], tasks: Sequence[str], expected: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(key) != expected_value for key, expected_value in identity.items()):
        raise ValueError(f"existing metric identity differs: {path}")
    metrics = value.get("metrics")
    if value.get("status") != "completed" or set(metrics or ()) != set(tasks):
        raise ValueError(f"existing metric grid differs: {path}")
    if any(metric.get("examples") != expected for metric in metrics.values()):
        raise ValueError(f"existing metric counts differ: {path}")
    return value


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    state_key: str,
    config_key: str,
    tasks: Sequence[str],
    spec: PropertyFewshotSpec,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainConfig.from_value(payload[config_key])
    model = _default_model_factory(config)
    model.load_state_dict(payload[state_key], strict=True)
    del payload
    model.to(device)
    dataset = StreamingPermutationDataset(
        spec.test_manifest,
        tasks=tasks,
        shard_indices=(spec.test_shard_position,),
        shuffle_buffer_size=1,
        seed=0,
        rank=0,
        world_size=1,
    )
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and spec.bf16 and torch.cuda.is_bf16_supported() else torch.float16
    metrics = evaluate_records(
        model,
        dataset,
        task_names=tasks,
        device=device,
        max_seq_len=config.max_seq_len,
        max_examples=32,
        max_padded_tokens=8192,
        amp_enabled=use_amp,
        amp_dtype=amp_dtype,
    )
    if any(metric["examples"] != spec.expected_test_examples for metric in metrics.values()):
        raise ValueError("source shard 199 evaluation count differs")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def evaluate_test(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    spec = load_spec(config_path)
    evaluator_commit = _git_commit(spec.repository)
    audit = audit_all(config_path)
    if not audit["ok"]:
        raise ValueError("all 144 Property32 few-shot runs must pass before test")
    verification = verify_manifest(spec.test_manifest, full=True, workers=1)
    if not verification["ok"] or verification["record_count"] != 160_000:
        raise ValueError("Property32 test manifest failed full verification")
    selected_shard = resolve_shards(spec.test_manifest, (spec.test_shard_position,))[0]
    if _sha256(selected_shard) != spec.test_source_shard_sha256:
        raise ValueError("source shard 199 changed before final evaluation")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_runs = _base_runs(spec, strict=False)
    plan = build_plan(spec, base_runs)
    lock_path = spec.evaluation_dir / ".evaluation.lock"
    descriptor = _locked(lock_path)
    os.close(descriptor)
    zero_results = []
    adapted_results = []
    try:
        for index, run in enumerate(base_runs, start=1):
            tasks = spec.target_sets[(str(run["replicate_id"]), str(run["model_pool"]))]
            identity = _metric_identity(
                format_version=ZERO_SHOT_FORMAT_VERSION,
                run_id=str(run["run_id"]),
                checkpoint_sha256=str(run["checkpoint_sha256"]),
                spec=spec,
                evaluator_commit=evaluator_commit,
            )
            output = spec.evaluation_dir / "zero-shot" / f"{run['run_id']}.json"
            result = _load_metric_result(output, identity, tasks, spec.expected_test_examples)
            if result is None:
                print(f"property zero-shot test {index}/30: {run['run_id']}", flush=True)
                metrics = _evaluate_checkpoint(
                    Path(str(run["checkpoint_path"])),
                    state_key="model",
                    config_key="config",
                    tasks=tasks,
                    spec=spec,
                    device=device,
                )
                result = {**identity, "status": "completed", "metrics": metrics}
                _atomic_json(result, output)
            zero_results.append(result)

        for index, run in enumerate(plan, start=1):
            run_dir = spec.output_dir / str(run["run_id"])
            marker = json.loads((run_dir / "completed.json").read_text(encoding="utf-8"))
            identity = {
                **_metric_identity(
                    format_version=TEST_FORMAT_VERSION,
                    run_id=str(run["run_id"]),
                    checkpoint_sha256=str(marker["checkpoint_sha256"]),
                    spec=spec,
                    evaluator_commit=evaluator_commit,
                ),
                "initialization": run["initialization"],
                "replicate_id": run["replicate_id"],
                "model_pool": run["model_pool"],
                "base_run_id": run["base_run_id"],
                "base_trained_task_count": run["base_trained_task_count"],
                "seed": run["seed"],
                "task": run["task"],
                "target_family": run["target_family"],
            }
            output = spec.evaluation_dir / "per-run" / f"{run['run_id']}.json"
            result = _load_metric_result(output, identity, (str(run["task"]),), spec.expected_test_examples)
            if result is None:
                print(f"property few-shot test {index}/144: {run['run_id']}", flush=True)
                metrics = _evaluate_checkpoint(
                    run_dir / "checkpoint.pt",
                    state_key="model",
                    config_key="base_train_config",
                    tasks=(str(run["task"]),),
                    spec=spec,
                    device=device,
                )
                result = {**identity, "status": "completed", "metrics": metrics}
                _atomic_json(result, output)
            adapted_results.append(result)
        manifest = {
            "format_version": TEST_FORMAT_VERSION,
            "status": "completed",
            "evaluator_commit": evaluator_commit,
            "fewshot_config_sha256": spec.config_sha256,
            "support_artifact_sha256": _sha256(spec.support_artifact),
            "test_manifest_sha256": spec.test_manifest_sha256,
            "test_shard_position": spec.test_shard_position,
            "test_source_shard_index": spec.test_source_shard_index,
            "test_source_shard_sha256": spec.test_source_shard_sha256,
            "full_dataset_verification": verification,
            "zero_shot_model_count": len(zero_results),
            "adapted_run_count": len(adapted_results),
            "examples_per_task": spec.expected_test_examples,
            "zero_shot_results": [f"zero-shot/{value['run_id']}.json" for value in zero_results],
            "adapted_results": [f"per-run/{value['run_id']}.json" for value in adapted_results],
        }
        _atomic_json(manifest, spec.evaluation_dir / "manifest.json")
        return manifest
    finally:
        lock_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("support", "plan", "run", "audit", "test"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--overwrite-support", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_spec(args.config)
    if args.command == "support":
        artifact = build_support_artifact(spec, overwrite=args.overwrite_support)
        output: Mapping[str, Any] = {"status": "completed", "set_count": artifact["set_count"], "record_count": artifact["record_count"]}
    elif args.command == "plan":
        plan = build_plan(spec, _base_runs(spec))
        output = {
            "status": "planned",
            "run_count": len(plan),
            "pretrained": sum(run["initialization"] == "pretrained" for run in plan),
            "random": sum(run["initialization"] == "random" for run in plan),
        }
    elif args.command == "run":
        output = run_all(args.config, device_name=args.device)
    elif args.command == "audit":
        output = audit_all(args.config)
    else:
        output = evaluate_test(args.config, device_name=args.device)
    print(json.dumps(dict(output), sort_keys=True, allow_nan=False))
    return 0 if output.get("status") in {"completed", "planned", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "FORMAT_VERSION",
    "PropertyFewshotSpec",
    "audit_all",
    "build_plan",
    "build_support_artifact",
    "evaluate_test",
    "load_spec",
    "load_support_artifact",
    "main",
    "run_all",
]

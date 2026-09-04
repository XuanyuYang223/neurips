"""Four-representation by eight-task transfer experiment.

The frozen design follows the supplied Passage Math specification: train one
joint Transformer on the complete one-line row and the descents column (11
unique cells), then evaluate all 32 representation-task cells.  The remaining
21 cells are held out from gradient updates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import sys
import tempfile
import time
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import math_ops as ops
from .evaluate import _atomic_json, _git_commit, evaluate_records
from .passage import REPRESENTATIONS, TOKEN_TO_ID, passage_tokens
from .training import StreamingPermutationDataset, TrainConfig, _default_model_factory, train_run


DEFAULT_CONFIG = Path("configs/permutation_representation_transfer.toml")
SCHEMA_VERSION = "permutation-representation-transfer-32/v1"
BASE_TASKS: tuple[str, ...] = (
    "length",
    "parity",
    "peaks",
    "exceedances",
    "fixed_points",
    "descents",
    "recoils",
    "lis_length",
)
ANCHOR_REPRESENTATION = "one_line"
ANCHOR_TASK = "descents"
FULL_COMBINATIONS: tuple[str, ...] = tuple(
    f"{representation}:{task}"
    for representation in REPRESENTATIONS
    for task in BASE_TASKS
)
TRAIN_COMBINATIONS: tuple[str, ...] = tuple(
    combination
    for combination in FULL_COMBINATIONS
    if combination.startswith(f"{ANCHOR_REPRESENTATION}:")
    or combination.endswith(f":{ANCHOR_TASK}")
)
HELD_OUT_COMBINATIONS: tuple[str, ...] = tuple(
    combination for combination in FULL_COMBINATIONS if combination not in TRAIN_COMBINATIONS
)
TASK_FUNCTIONS = {
    "length": ops.inversion_count,
    "parity": ops.parity,
    "peaks": ops.peak_count,
    "exceedances": ops.exceedance_count,
    "fixed_points": ops.fixed_point_count,
    "descents": ops.descent_count,
    "recoils": ops.recoil_count,
    "lis_length": ops.lis_length,
}
SPLITS = ("train", "validation", "test")
GZIP_LEVEL = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path, *, require_derived: bool = True) -> tuple[dict[str, Any], Path, str]:
    config_path = path.resolve()
    repository = config_path.parent.parent
    payload = config_path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    if config.get("protocol_version") != "permutation-representation-transfer/v1":
        raise ValueError("unsupported representation-transfer protocol")
    if tuple(config.get("representations", ())) != REPRESENTATIONS:
        raise ValueError("representation order differs from the frozen protocol")
    if tuple(config.get("tasks", ())) != BASE_TASKS:
        raise ValueError("task order differs from the frozen protocol")
    if tuple(config.get("training_combinations", ())) != TRAIN_COMBINATIONS:
        raise ValueError("training grid differs from the frozen row-plus-column design")
    if tuple(config.get("held_out_combinations", ())) != HELD_OUT_COMBINATIONS:
        raise ValueError("held-out grid differs from the frozen protocol")
    if tuple(config.get("model_seeds", ())) != (17, 42, 314159):
        raise ValueError("the protocol requires the three frozen model seeds")
    source = config.get("source_artifact")
    if not isinstance(source, dict):
        raise ValueError("source_artifact must be a TOML table")
    for split in SPLITS:
        source_path = repository / str(config[f"source_{split}_manifest"])
        if _sha256(source_path) != source.get(f"{split}_manifest_sha256"):
            raise ValueError(f"source {split} manifest differs from the frozen artifact")
    if require_derived:
        derived = config.get("derived_artifact")
        if not isinstance(derived, dict):
            raise ValueError("derived_artifact must be frozen after data preparation")
        for split in SPLITS:
            manifest = repository / str(config[f"{split}_manifest"])
            if _sha256(manifest) != derived.get(f"{split}_manifest_sha256"):
                raise ValueError(f"derived {split} manifest differs from the frozen artifact")
    return config, repository, hashlib.sha256(payload).hexdigest()


def _source_shards(manifest_path: Path) -> tuple[tuple[int, Path], ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("shards")
    if not isinstance(entries, list):
        raise ValueError(f"source manifest has no shards: {manifest_path}")
    result = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("filename"), str):
            raise ValueError("source shard entry is invalid")
        index = int(entry.get("index", position))
        path = manifest_path.parent / str(entry["filename"])
        if not path.is_file() or _sha256(path) != entry.get("sha256"):
            raise ValueError(f"source shard does not match its manifest: {path}")
        result.append((index, path))
    return tuple(result)


def combinations_for_split(split: str) -> tuple[str, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    return TRAIN_COMBINATIONS if split == "train" else FULL_COMBINATIONS


def _representations_for_record(split: str, task: str) -> tuple[str, ...]:
    if split != "train":
        return REPRESENTATIONS
    if task == ANCHOR_TASK:
        return REPRESENTATIONS
    return (ANCHOR_REPRESENTATION,)


def _record(source: Mapping[str, Any], representation: str) -> dict[str, Any]:
    task = str(source["task"])
    primary = tuple(int(value) for value in source["inputs"]["primary"])
    answer = int(TASK_FUNCTIONS[task](primary))
    tokens = passage_tokens(task, primary, answer, representation=representation)
    representation_index = REPRESENTATIONS.index(representation)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": int(source["id"]) * len(REPRESENTATIONS) + representation_index,
        "source_id": int(source["id"]),
        "task": f"{representation}:{task}",
        "base_task": task,
        "representation": representation,
        "n": len(primary),
        "inputs": {"primary": list(primary)},
        "answer": answer,
        "answer_kind": "scalar",
        "tokens": list(tokens),
        "canonical_text": " ".join(tokens),
    }


@dataclass(frozen=True)
class _DeriveJob:
    split: str
    source_index: int
    output_index: int
    source_path: str
    destination: str


def _derive_shard(job: _DeriveJob) -> dict[str, Any]:
    destination = Path(job.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    counts: Counter[str] = Counter()
    first_id: int | None = None
    last_id: int | None = None
    record_count = 0
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=GZIP_LEVEL, mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                    with gzip.open(job.source_path, "rt", encoding="utf-8") as source:
                        for line in source:
                            value = json.loads(line)
                            task = value.get("task")
                            if task not in BASE_TASKS:
                                continue
                            for representation in _representations_for_record(job.split, task):
                                record = _record(value, representation)
                                output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
                                output.write("\n")
                                record_id = int(record["id"])
                                first_id = record_id if first_id is None else min(first_id, record_id)
                                last_id = record_id if last_id is None else max(last_id, record_id)
                                counts[str(record["task"])] += 1
                                record_count += 1
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {
        "index": job.output_index,
        "source_shard_index": job.source_index,
        "filename": destination.name,
        "record_count": record_count,
        "first_id": first_id,
        "last_id": last_id,
        "byte_size": destination.stat().st_size,
        "sha256": _sha256(destination),
        "task_counts": dict(sorted(counts.items())),
    }


def derive_split(
    config_path: Path,
    split: str,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    config, repository, _ = _read_config(config_path, require_derived=False)
    if workers < 1:
        raise ValueError("workers must be positive")
    source_manifest = repository / str(config[f"source_{split}_manifest"])
    output_root = repository / str(config["data_output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for output_index, (source_index, source_path) in enumerate(_source_shards(source_manifest)):
        jobs.append(
            _DeriveJob(
                split=split,
                source_index=source_index,
                output_index=output_index,
                source_path=str(source_path),
                destination=str(output_root / f"{split}-part-{output_index:05d}.jsonl.gz"),
            )
        )
    results = []
    if workers == 1:
        results = [_derive_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            pending = {executor.submit(_derive_shard, job): job for job in jobs}
            for future in as_completed(pending):
                results.append(future.result())
    results.sort(key=lambda entry: int(entry["index"]))
    task_counts: Counter[str] = Counter()
    for entry in results:
        task_counts.update(entry["task_counts"])
    combinations = combinations_for_split(split)
    expected_per_cell = 490_000 if split == "train" else 5_000
    expected_counts = {combination: expected_per_cell for combination in combinations}
    if dict(sorted(task_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(f"derived {split} task counts are not exactly balanced")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": config["protocol_version"],
        "split": split,
        "source_manifest": str(config[f"source_{split}_manifest"]),
        "source_manifest_sha256": _sha256(source_manifest),
        "representations": list(REPRESENTATIONS),
        "base_tasks": list(BASE_TASKS),
        "tasks": list(combinations),
        "training_combinations": list(TRAIN_COMBINATIONS),
        "held_out_combinations": list(HELD_OUT_COMBINATIONS),
        "count": sum(task_counts.values()),
        "task_counts": dict(sorted(task_counts.items())),
        "shard_count": len(results),
        "total_bytes": sum(int(entry["byte_size"]) for entry in results),
        "shards": results,
    }
    destination = repository / str(config[f"{split}_manifest"])
    _atomic_json(manifest, destination)
    return manifest


def prepare_datasets(config_path: Path = DEFAULT_CONFIG, *, workers: int = 8) -> dict[str, Any]:
    results = {split: derive_split(config_path, split, workers=workers) for split in SPLITS}
    _, repository, _ = _read_config(config_path, require_derived=False)
    return {
        split: {
            "records": result["count"],
            "bytes": result["total_bytes"],
            "manifest_sha256": _sha256(
                repository
                / str(tomllib.loads(config_path.read_text(encoding="utf-8"))[f"{split}_manifest"])
            ),
        }
        for split, result in results.items()
    }


def verify_manifest(path: Path, *, full: bool = True) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("not a representation-transfer manifest")
    split = str(manifest.get("split"))
    allowed = combinations_for_split(split)
    if manifest.get("tasks") != list(allowed):
        raise ValueError("manifest task registry is invalid")
    counts: Counter[str] = Counter()
    records = 0
    bytes_total = 0
    for entry in manifest.get("shards", ()):
        shard = path.parent / str(entry["filename"])
        if shard.stat().st_size != entry["byte_size"] or _sha256(shard) != entry["sha256"]:
            raise ValueError(f"shard differs from manifest: {shard}")
        bytes_total += shard.stat().st_size
        if not full:
            counts.update(entry["task_counts"])
            records += int(entry["record_count"])
            continue
        shard_records = 0
        shard_counts: Counter[str] = Counter()
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                task = value.get("base_task")
                representation = value.get("representation")
                combination = value.get("task")
                if task not in BASE_TASKS or representation not in REPRESENTATIONS:
                    raise ValueError(f"invalid record registry in {shard}")
                if combination != f"{representation}:{task}" or combination not in allowed:
                    raise ValueError(f"record is outside the {split} grid")
                primary = tuple(value["inputs"]["primary"])
                truth = int(TASK_FUNCTIONS[task](primary))
                tokens = passage_tokens(task, primary, truth, representation=representation)
                if value.get("answer") != truth:
                    raise ValueError("stored answer differs from mathematical truth")
                if value.get("tokens") != list(tokens) or value.get("canonical_text") != " ".join(tokens):
                    raise ValueError("stored Passage Math sequence is not canonical")
                if value.get("id") != int(value["source_id"]) * 4 + REPRESENTATIONS.index(representation):
                    raise ValueError("derived record ID is invalid")
                shard_counts[combination] += 1
                shard_records += 1
        if shard_records != entry["record_count"] or dict(sorted(shard_counts.items())) != entry["task_counts"]:
            raise ValueError("shard counts differ from manifest")
        counts.update(shard_counts)
        records += shard_records
    if records != manifest.get("count") or dict(sorted(counts.items())) != manifest.get("task_counts"):
        raise ValueError("global counts differ from manifest")
    if bytes_total != manifest.get("total_bytes"):
        raise ValueError("global byte count differs from manifest")
    return {
        "ok": True,
        "full": full,
        "split": split,
        "record_count": records,
        "task_counts": dict(sorted(counts.items())),
        "total_bytes": bytes_total,
    }


@dataclass(frozen=True)
class RepresentationRun:
    run_id: str
    seed: int
    output_dir: str


def build_runs(config_path: Path = DEFAULT_CONFIG) -> tuple[RepresentationRun, ...]:
    config, _, _ = _read_config(config_path)
    root = Path(str(config["run_output_dir"]))
    return tuple(
        RepresentationRun(
            run_id=f"representation-transfer-transformer-seed{seed}",
            seed=int(seed),
            output_dir=str(root / f"representation-transfer-transformer-seed{seed}"),
        )
        for seed in config["model_seeds"]
    )


def _train_config(run: RepresentationRun, config_path: Path) -> TrainConfig:
    config, _, config_sha = _read_config(config_path)
    model = config["model"]
    training = config["training"]
    data = config["data"]
    return TrainConfig(
        output_dir=run.output_dir,
        manifest=str(config["train_manifest"]),
        validation_manifest=str(config["validation_manifest"]),
        architecture="transformer",
        d_model=int(model["d_model"]),
        num_layers=int(model["layers"]),
        num_heads=int(model["num_heads"]),
        dropout=float(model["dropout"]),
        mlp_ratio=float(model["ff_multiplier"]),
        tie_embeddings=bool(model["tie_embeddings"]),
        model_config={"vocab_size": len(TOKEN_TO_ID)},
        tasks=TRAIN_COMBINATIONS,
        validation_tasks=FULL_COMBINATIONS,
        seed=run.seed,
        max_steps=int(training["max_steps"]),
        batch_size=int(training["micro_batch_size"]),
        validation_batch_size=int(training["validation_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        max_seq_len=int(data["max_sequence_length"]),
        max_tokens_per_batch=int(training["max_tokens_per_batch"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_steps=int(training["warmup_steps"]),
        min_lr_ratio=float(training["min_learning_rate_ratio"]),
        max_grad_norm=float(training["gradient_clip_norm"]),
        shuffle_buffer_size=int(data["shuffle_buffer"]),
        num_workers=int(training["num_workers"]),
        checkpoint_every=int(training["checkpoint_every_steps"]),
        validate_every=int(training["validate_every_steps"]),
        validation_batches_per_task=int(training["validation_batches_per_task"]),
        amp=True,
        bf16=True,
        resume="auto",
        experiment_config=str(config_path),
        experiment_config_sha256=config_sha,
    )


def _completed(run: RepresentationRun, expected: TrainConfig) -> bool:
    marker_path = Path(run.output_dir) / "completed.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        checkpoint = Path(run.output_dir) / "checkpoint.pt"
        return (
            marker.get("status") == "completed"
            and marker.get("global_step") == expected.max_steps
            and marker.get("run_id") == run.run_id
            and marker.get("seed") == run.seed
            and marker.get("tasks") == list(TRAIN_COMBINATIONS)
            and set(marker.get("validation", ())) == set(FULL_COMBINATIONS)
            and checkpoint.is_file()
            and marker.get("checkpoint_sha256") == _sha256(checkpoint)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def status(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    runs = build_runs(config_path)
    rows = []
    for run in runs:
        expected = _train_config(run, config_path)
        rows.append({**asdict(run), "completed": _completed(run, expected)})
    return {
        "run_count": len(rows),
        "completed_count": sum(bool(row["completed"]) for row in rows),
        "incomplete_count": sum(not row["completed"] for row in rows),
        "runs": rows,
    }


def train_one(config_path: Path, run_id: str) -> dict[str, Any]:
    run = next((run for run in build_runs(config_path) if run.run_id == run_id), None)
    if run is None:
        raise ValueError(f"unknown run_id {run_id!r}")
    return train_run(_train_config(run, config_path))


def run_matrix(config_path: Path = DEFAULT_CONFIG, *, dry_run: bool = False) -> None:
    for run in build_runs(config_path):
        expected = _train_config(run, config_path)
        if _completed(run, expected):
            continue
        command = [
            sys.executable,
            "-m",
            "neurips_permutations.representation_transfer",
            "train-one",
            "--config",
            str(config_path),
            "--run-id",
            run.run_id,
        ]
        if dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)


def audit_runs(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    rows = []
    for run in build_runs(config_path):
        expected = _train_config(run, config_path)
        issues = []
        checkpoint_path = Path(run.output_dir) / "checkpoint.pt"
        if not _completed(run, expected):
            issues.append("completion marker or checkpoint identity is invalid")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("config") != asdict(expected):
                issues.append("checkpoint TrainConfig differs from the frozen protocol")
            if checkpoint.get("state", {}).get("global_step") != expected.max_steps:
                issues.append("checkpoint step is incomplete")
            for section in ("model", "optimizer"):
                stack: list[Any] = [checkpoint.get(section)]
                while stack:
                    value = stack.pop()
                    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                        issues.append(f"{section} contains non-finite tensors")
                        break
                    if isinstance(value, Mapping):
                        stack.extend(value.values())
                    elif isinstance(value, (list, tuple)):
                        stack.extend(value)
        rows.append({"run_id": run.run_id, "ok": not issues, "issues": issues})
    return {
        "ok": all(row["ok"] for row in rows),
        "passed_count": sum(row["ok"] for row in rows),
        "failed_count": sum(not row["ok"] for row in rows),
        "runs": rows,
    }


def evaluate_test(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    config, repository, _ = _read_config(config_path)
    audit = audit_runs(config_path)
    if not audit["ok"]:
        raise ValueError("all representation-transfer checkpoints must pass strict audit")
    test_manifest = repository / str(config["test_manifest"])
    verification = verify_manifest(test_manifest, full=True)
    evaluator_commit = _git_commit(repository)
    output_dir = repository / str(config["result_output_dir"]) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".evaluation.lock"
    try:
        lock.open("x").close()
    except FileExistsError as error:
        raise RuntimeError("representation-transfer test evaluator is already active") from error
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    outputs = []
    try:
        for index, run in enumerate(build_runs(config_path), start=1):
            checkpoint_path = Path(run.output_dir) / "checkpoint.pt"
            identity = {
                "format_version": "permutation-representation-transfer-test/v1",
                "run_id": run.run_id,
                "seed": run.seed,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "test_manifest_sha256": _sha256(test_manifest),
                "evaluator_commit": evaluator_commit,
            }
            destination = output_dir / f"{run.run_id}.json"
            if destination.is_file():
                result = json.loads(destination.read_text(encoding="utf-8"))
                if any(result.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"existing test result identity differs: {destination}")
                outputs.append(result)
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            train_config = TrainConfig.from_value(checkpoint["config"])
            model = _default_model_factory(train_config)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            dataset = StreamingPermutationDataset(
                test_manifest,
                tasks=FULL_COMBINATIONS,
                shuffle_buffer_size=1,
                seed=run.seed,
            )
            use_amp = train_config.amp and device.type == "cuda"
            amp_dtype = torch.bfloat16 if use_amp and train_config.bf16 else torch.float16
            print(f"representation transfer test {index}/3: {run.run_id}", flush=True)
            started = time.monotonic()
            metrics = evaluate_records(
                model,
                dataset,
                task_names=FULL_COMBINATIONS,
                device=device,
                max_seq_len=train_config.max_seq_len,
                max_examples=int(config["evaluation"]["max_examples_per_batch"]),
                max_padded_tokens=int(config["evaluation"]["max_padded_tokens"]),
                amp_enabled=use_amp,
                amp_dtype=amp_dtype,
            )
            if any(metric["examples"] != 5_000 for metric in metrics.values()):
                raise ValueError("test evaluation must contain 5,000 examples per cell")
            result = {
                **identity,
                "status": "completed",
                "elapsed_seconds": time.monotonic() - started,
                "metrics": metrics,
            }
            _atomic_json(result, destination)
            outputs.append(result)
            del checkpoint, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        manifest = {
            "status": "completed",
            "format_version": "permutation-representation-transfer-test/v1",
            "run_count": len(outputs),
            "cell_count": len(FULL_COMBINATIONS),
            "examples_per_cell_per_run": 5_000,
            "total_model_examples": 3 * 32 * 5_000,
            "test_verification": verification,
            "runs": [
                {key: value for key, value in result.items() if key != "metrics"}
                for result in outputs
            ],
        }
        _atomic_json(manifest, output_dir / "manifest.json")
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    values = tuple(values)
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    import csv

    config, repository, _ = _read_config(config_path)
    evaluation_dir = repository / str(config["result_output_dir"]) / "evaluation"
    results = [
        json.loads((evaluation_dir / f"{run.run_id}.json").read_text(encoding="utf-8"))
        for run in build_runs(config_path)
    ]
    raw = []
    for result in results:
        for combination in FULL_COMBINATIONS:
            representation, task = combination.split(":", 1)
            metric = result["metrics"][combination]
            raw.append(
                {
                    "run_id": result["run_id"],
                    "seed": result["seed"],
                    "representation": representation,
                    "task": task,
                    "combination": combination,
                    "task_status": "seen" if combination in TRAIN_COMBINATIONS else "held_out",
                    "examples": metric["examples"],
                    "supervised_tokens": metric["tokens"],
                    "loss": metric["loss"],
                    "token_accuracy": metric["token_accuracy"],
                    "sequence_accuracy": metric["sequence_accuracy"],
                }
            )
    output_dir = repository / str(config["result_output_dir"])
    fields = tuple(raw[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / ".raw.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw)
    os.replace(temporary, output_dir / "MODEL_CELL_ACCURACIES.csv")
    summary = []
    for status_name in ("seen", "held_out"):
        per_seed = []
        for seed in config["model_seeds"]:
            rows = [row for row in raw if row["seed"] == seed and row["task_status"] == status_name]
            per_seed.append(
                (
                    statistics.fmean(float(row["loss"]) for row in rows),
                    statistics.fmean(float(row["token_accuracy"]) for row in rows),
                    statistics.fmean(float(row["sequence_accuracy"]) for row in rows),
                )
            )
        summary.append(
            {
                "task_status": status_name,
                "cell_count": 11 if status_name == "seen" else 21,
                "loss_mean": statistics.fmean(value[0] for value in per_seed),
                "loss_sample_sd": statistics.stdev(value[0] for value in per_seed),
                "token_accuracy_mean": statistics.fmean(value[1] for value in per_seed),
                "token_accuracy_sample_sd": statistics.stdev(value[1] for value in per_seed),
                "sequence_accuracy_mean": statistics.fmean(value[2] for value in per_seed),
                "sequence_accuracy_sample_sd": statistics.stdev(value[2] for value in per_seed),
            }
        )
    _atomic_json({"raw_row_count": len(raw), "summary": summary}, output_dir / "summary.json")
    lines = [
        "# Four-representation transfer results",
        "",
        "Three jointly trained Transformers saw the one-line row and descents column (11 cells).",
        "The other 21 representation-task combinations received no gradient updates.",
        "",
        "| Status | Cells | Loss | Token accuracy | Exact-sequence accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['task_status']} | {row['cell_count']} | "
            f"{row['loss_mean']:.3f} ± {row['loss_sample_sd']:.3f} | "
            f"{100*row['token_accuracy_mean']:.2f} ± {100*row['token_accuracy_sample_sd']:.2f}% | "
            f"{100*row['sequence_accuracy_mean']:.2f} ± {100*row['sequence_accuracy_sample_sd']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Means are cell-macro averages computed within each seed, followed by mean ± sample SD over three seeds.",
            "Exact-sequence accuracy is the primary complete-answer metric; token accuracy is teacher-forced.",
            "",
        ]
    )
    destination = output_dir / "README.md"
    temporary = output_dir / ".README.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)
    return {"raw_row_count": len(raw), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify", "plan", "status", "run", "audit", "test", "report"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if command in {"prepare", "verify"}:
            sub.add_argument("--workers", type=int, default=8)
        if command == "verify":
            sub.add_argument("--full", action="store_true")
        if command == "run":
            sub.add_argument("--dry-run", action="store_true")
        if command == "test":
            sub.add_argument("--device")
    train_one_parser = subparsers.add_parser("train-one")
    train_one_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train_one_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        value = prepare_datasets(args.config, workers=args.workers)
    elif args.command == "verify":
        config, repository, _ = _read_config(args.config)
        value = {
            split: verify_manifest(repository / str(config[f"{split}_manifest"]), full=args.full)
            for split in SPLITS
        }
    elif args.command in {"plan", "status"}:
        value = status(args.config)
    elif args.command == "run":
        run_matrix(args.config, dry_run=args.dry_run)
        return 0
    elif args.command == "train-one":
        value = train_one(args.config, args.run_id)
    elif args.command == "audit":
        value = audit_runs(args.config)
    elif args.command == "test":
        value = evaluate_test(args.config, device_name=args.device)
    else:
        value = report(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BASE_TASKS",
    "FULL_COMBINATIONS",
    "HELD_OUT_COMBINATIONS",
    "SCHEMA_VERSION",
    "TRAIN_COMBINATIONS",
    "audit_runs",
    "build_runs",
    "combinations_for_split",
    "derive_split",
    "evaluate_test",
    "main",
    "prepare_datasets",
    "report",
    "run_matrix",
    "status",
    "verify_manifest",
]

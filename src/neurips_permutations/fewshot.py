"""Henry-style 20-shot adaptation and random-initialization controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import tomllib
from typing import Any, Mapping, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .audit import audit_experiment
from .evaluate import evaluate_records
from .training import (
    AnswerOnlyCollator,
    StreamingPermutationDataset,
    TrainConfig,
    _autocast_context,
    _default_model_factory,
    _model_logits,
    _causal_loss,
    resolve_shards,
)
from .verify import verify_manifest


DEFAULT_CONFIG = Path("configs/henry_permutation_fewshot.toml")
FORMAT_VERSION = "henry-permutation-fewshot/v1"
SUPPORT_FORMAT_VERSION = "henry-permutation-fewshot-support/v1"
TEST_FORMAT_VERSION = "henry-permutation-fewshot-test/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, destination)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, destination)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _portable_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def _git_commit(repository: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("git HEAD is not a full commit hash")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(["git", *args], cwd=repository, check=False)
        if completed.returncode:
            raise RuntimeError("tracked worktree must be clean before a formal run")
    return commit


@dataclass(frozen=True)
class FewshotSpec:
    repository: Path
    config_path: Path
    config_sha256: str
    base_config_path: Path
    base_config_sha256: str
    train_manifest: Path
    train_manifest_sha256: str
    validation_manifest: Path
    validation_manifest_sha256: str
    test_manifest: Path
    test_manifest_sha256: str
    support_artifact: Path
    output_dir: Path
    evaluation_dir: Path
    holdout_tasks: tuple[str, ...]
    model_seeds: tuple[int, ...]
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


def load_spec(config_path: Path = DEFAULT_CONFIG) -> FewshotSpec:
    config_path = config_path.resolve()
    repository = config_path.parent.parent
    payload = config_path.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))

    def path(name: str) -> Path:
        return (repository / str(config[name])).resolve()

    fine = config["fine_tuning"]
    evaluation = config["evaluation"]
    matrix = config["matrix"]
    spec = FewshotSpec(
        repository=repository,
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        base_config_path=path("base_experiment_config"),
        base_config_sha256=str(config["base_experiment_config_sha256"]),
        train_manifest=path("train_manifest"),
        train_manifest_sha256=str(config["train_manifest_sha256"]),
        validation_manifest=path("validation_manifest"),
        validation_manifest_sha256=str(config["validation_manifest_sha256"]),
        test_manifest=path("test_manifest"),
        test_manifest_sha256=str(config["test_manifest_sha256"]),
        support_artifact=path("support_artifact"),
        output_dir=path("output_dir"),
        evaluation_dir=path("evaluation_dir"),
        holdout_tasks=tuple(config["holdout_tasks"]),
        model_seeds=tuple(int(seed) for seed in config["model_seeds"]),
        shots=int(config["shots"]),
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
        expected_runs=int(matrix["total_runs"]),
    )
    if config.get("protocol_version") != FORMAT_VERSION:
        raise ValueError("unsupported few-shot protocol version")
    if len(spec.holdout_tasks) != 4 or len(set(spec.holdout_tasks)) != 4:
        raise ValueError("few-shot protocol requires four distinct holdout tasks")
    if len(spec.model_seeds) != 3 or len(set(spec.model_seeds)) != 3:
        raise ValueError("few-shot protocol requires three distinct model seeds")
    if spec.shots != 20:
        raise ValueError("Henry's primary few-shot protocol is frozen at 20 shots")
    for value in (
        spec.max_steps,
        spec.batch_size,
        spec.gradient_accumulation_steps,
        spec.max_tokens_per_batch,
        spec.expected_validation_examples,
        spec.expected_test_examples,
    ):
        if value < 1:
            raise ValueError("few-shot integer settings must be positive")
    if (
        spec.learning_rate <= 0
        or spec.random_init_learning_rate <= 0
        or spec.weight_decay < 0
        or spec.max_grad_norm <= 0
    ):
        raise ValueError("invalid few-shot optimizer settings")
    if not 0 <= spec.min_learning_rate_ratio <= 1:
        raise ValueError("invalid few-shot minimum learning-rate ratio")
    if _sha256(spec.base_config_path) != spec.base_config_sha256:
        raise ValueError("base experiment config differs from the frozen few-shot spec")
    for manifest, expected in (
        (spec.train_manifest, spec.train_manifest_sha256),
        (spec.validation_manifest, spec.validation_manifest_sha256),
        (spec.test_manifest, spec.test_manifest_sha256),
    ):
        if _sha256(manifest) != expected:
            raise ValueError(f"manifest differs from frozen few-shot spec: {manifest}")
    return spec


def _support_key(task: str, seed: int) -> str:
    return f"{task}:seed{seed}"


def build_support_artifact(spec: FewshotSpec, *, overwrite: bool = False) -> dict[str, Any]:
    """Select paired 20-shot support sets from train shards only."""

    if spec.support_artifact.exists() and not overwrite:
        return load_support_artifact(spec)
    manifest = json.loads(spec.train_manifest.read_text(encoding="utf-8"))
    task_names = tuple(manifest["tasks"])
    if not set(spec.holdout_tasks) <= set(task_names):
        raise ValueError("holdout tasks are absent from the training manifest")
    if manifest.get("count") != 9_800_000 or len(manifest.get("shards", ())) != 98:
        raise ValueError("support examples must come from the frozen 9.8M train split")
    task_count = len(task_names)
    desired: dict[int, tuple[str, int]] = {}
    set_ids: dict[str, list[int]] = {}
    for seed in spec.model_seeds:
        for task in spec.holdout_tasks:
            available = int(manifest["task_counts"][task])
            rng = random.Random(_stable_seed(FORMAT_VERSION, "support", task, seed))
            occurrences = sorted(rng.sample(range(available), spec.shots))
            task_index = task_names.index(task)
            ids = [task_index + task_count * occurrence for occurrence in occurrences]
            key = _support_key(task, seed)
            set_ids[key] = ids
            for record_id in ids:
                if record_id in desired:
                    raise ValueError("support sample collision")
                desired[record_id] = (task, seed)

    found: dict[int, dict[str, Any]] = {}
    for shard in resolve_shards(spec.train_manifest):
        entry = next(
            item for item in manifest["shards"] if item["filename"] == shard.name
        )
        wanted = {
            record_id
            for record_id in desired
            if int(entry["first_id"]) <= record_id <= int(entry["last_id"])
        }
        if not wanted:
            continue
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record_id = int(record["id"])
                if record_id in wanted:
                    task, _ = desired[record_id]
                    if record.get("task") != task:
                        raise ValueError("ID-to-task schedule differs in support data")
                    found[record_id] = record
        missing = wanted - set(found)
        if missing:
            raise ValueError(f"support records missing from shard {shard}: {missing}")
    if set(found) != set(desired):
        raise ValueError("not all selected support records were found")

    sets = []
    for seed in spec.model_seeds:
        for task in spec.holdout_tasks:
            key = _support_key(task, seed)
            ids = set_ids[key]
            sets.append(
                {
                    "key": key,
                    "task": task,
                    "seed": seed,
                    "record_ids": ids,
                    "records": [found[record_id] for record_id in ids],
                }
            )
    artifact = {
        "format_version": SUPPORT_FORMAT_VERSION,
        "fewshot_config_sha256": spec.config_sha256,
        "train_manifest_sha256": spec.train_manifest_sha256,
        "selection_method": "seeded_uniform_occurrence_without_replacement/v1",
        "shots": spec.shots,
        "tasks": list(spec.holdout_tasks),
        "seeds": list(spec.model_seeds),
        "set_count": len(sets),
        "record_count": sum(len(item["records"]) for item in sets),
        "sets": sets,
    }
    _atomic_json(artifact, spec.support_artifact)
    return artifact


def load_support_artifact(spec: FewshotSpec) -> dict[str, Any]:
    artifact = json.loads(spec.support_artifact.read_text(encoding="utf-8"))
    expected = {
        "format_version": SUPPORT_FORMAT_VERSION,
        "fewshot_config_sha256": spec.config_sha256,
        "train_manifest_sha256": spec.train_manifest_sha256,
        "shots": spec.shots,
        "tasks": list(spec.holdout_tasks),
        "seeds": list(spec.model_seeds),
        "set_count": len(spec.holdout_tasks) * len(spec.model_seeds),
        "record_count": spec.shots * len(spec.holdout_tasks) * len(spec.model_seeds),
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("support artifact identity differs from the frozen spec")
    seen_ids: set[int] = set()
    for item in artifact.get("sets", ()):
        records = item.get("records")
        if len(records) != spec.shots or item.get("record_ids") != [
            record.get("id") for record in records
        ]:
            raise ValueError("support set has invalid records or IDs")
        if any(record.get("task") != item.get("task") for record in records):
            raise ValueError("support set contains the wrong task")
        ids = {int(record["id"]) for record in records}
        if len(ids) != spec.shots or seen_ids & ids:
            raise ValueError("support records must be unique across all paired sets")
        seen_ids.update(ids)
    return artifact


def _support_lookup(artifact: Mapping[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    return {
        (str(item["task"]), int(item["seed"])): list(item["records"])
        for item in artifact["sets"]
    }


def build_plan(spec: FewshotSpec, audited_runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build 120 warm-start adaptations plus 24 random controls."""

    if len(audited_runs) != 30:
        raise ValueError("few-shot plan requires all 30 nested base models")
    base_runs = sorted(
        audited_runs,
        key=lambda run: (
            str(run["architecture"]),
            int(run["task_count"]),
            spec.model_seeds.index(int(run["seed"])),
        ),
    )
    plan: list[dict[str, Any]] = []
    for run in base_runs:
        if run.get("status") != "passed":
            raise ValueError(f"base model is not strictly audited: {run['run_id']}")
        for task in spec.holdout_tasks:
            plan.append(
                {
                    "run_id": f"pretrained-{run['run_id']}-{task}",
                    "initialization": "pretrained",
                    "architecture": run["architecture"],
                    "base_run_id": run["run_id"],
                    "base_trained_task_count": int(run["task_count"]),
                    "seed": int(run["seed"]),
                    "task": task,
                    "source_checkpoint": run["checkpoint_path"],
                    "source_checkpoint_sha256": run["checkpoint_sha256"],
                    "model_config_checkpoint_sha256": run["checkpoint_sha256"],
                }
            )
    for architecture in ("transformer", "mlp"):
        for seed in spec.model_seeds:
            template = next(
                run
                for run in base_runs
                if run["architecture"] == architecture
                and int(run["seed"]) == seed
                and int(run["task_count"]) == 1
            )
            for task in spec.holdout_tasks:
                plan.append(
                    {
                        "run_id": f"random-{architecture}-seed{seed}-{task}",
                        "initialization": "random",
                        "architecture": architecture,
                        "base_run_id": "",
                        "base_trained_task_count": 0,
                        "seed": seed,
                        "task": task,
                        "source_checkpoint": template["checkpoint_path"],
                        "source_checkpoint_sha256": "random_init",
                        "model_config_checkpoint_sha256": template[
                            "checkpoint_sha256"
                        ],
                    }
                )
    if len(plan) != spec.expected_runs or len({run["run_id"] for run in plan}) != len(plan):
        raise ValueError("few-shot plan does not contain 144 unique runs")
    return plan


def _run_identity(
    spec: FewshotSpec,
    run: Mapping[str, Any],
    *,
    support_sha256: str,
    support_ids: Sequence[int],
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "run_id": run["run_id"],
        "initialization": run["initialization"],
        "architecture": run["architecture"],
        "base_run_id": run["base_run_id"],
        "base_trained_task_count": run["base_trained_task_count"],
        "seed": run["seed"],
        "task": run["task"],
        "shots": spec.shots,
        "support_record_ids": list(support_ids),
        "source_checkpoint_sha256": run["source_checkpoint_sha256"],
        "model_config_checkpoint_sha256": run["model_config_checkpoint_sha256"],
        "fewshot_config_sha256": spec.config_sha256,
        "support_artifact_sha256": support_sha256,
        "implementation_commit": implementation_commit,
        "max_steps": spec.max_steps,
        "batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": (
            spec.learning_rate
            if run["initialization"] == "pretrained"
            else spec.random_init_learning_rate
        ),
    }


def _completion_valid(path: Path, identity: Mapping[str, Any], checkpoint_path: Path) -> bool:
    if not path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if marker.get("status") != "completed":
        return False
    if any(marker.get(key) != value for key, value in identity.items()):
        return False
    checkpoint_sha256 = marker.get("checkpoint_sha256")
    return isinstance(checkpoint_sha256, str) and _sha256(checkpoint_path) == checkpoint_sha256


def _scheduler(optimizer: AdamW, spec: FewshotSpec) -> LambdaLR:
    def multiplier(step: int) -> float:
        if spec.warmup_steps and step < spec.warmup_steps:
            return max(1e-12, (step + 1) / spec.warmup_steps)
        span = max(1, spec.max_steps - spec.warmup_steps)
        progress = min(1.0, max(0.0, (step - spec.warmup_steps) / span))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return spec.min_learning_rate_ratio + (1 - spec.min_learning_rate_ratio) * cosine

    return LambdaLR(optimizer, multiplier)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one(
    spec: FewshotSpec,
    run: Mapping[str, Any],
    support_records: Sequence[Mapping[str, Any]],
    *,
    support_sha256: str,
    implementation_commit: str,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = spec.output_dir / str(run["run_id"])
    checkpoint_path = run_dir / "checkpoint.pt"
    marker_path = run_dir / "completed.json"
    support_ids = [int(record["id"]) for record in support_records]
    identity = _run_identity(
        spec,
        run,
        support_sha256=support_sha256,
        support_ids=support_ids,
        implementation_commit=implementation_commit,
    )
    if _completion_valid(marker_path, identity, checkpoint_path):
        return json.loads(marker_path.read_text(encoding="utf-8"))
    if len(support_records) != spec.shots or any(
        record.get("task") != run["task"] for record in support_records
    ):
        raise ValueError("adaptation support set is invalid")

    source_checkpoint = Path(str(run["source_checkpoint"]))
    if not source_checkpoint.is_absolute():
        source_checkpoint = spec.repository / source_checkpoint
    if _sha256(source_checkpoint) != run["model_config_checkpoint_sha256"]:
        raise ValueError("model-configuration source checkpoint changed")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    base_config = TrainConfig.from_value(source["config"])
    adaptation_seed = _stable_seed(FORMAT_VERSION, run["task"], run["seed"])
    _seed_everything(adaptation_seed)
    model = _default_model_factory(base_config)
    if run["initialization"] == "pretrained":
        if _sha256(source_checkpoint) != run["source_checkpoint_sha256"]:
            raise ValueError(f"base checkpoint changed: {run['base_run_id']}")
        model.load_state_dict(source["model"], strict=True)
    del source
    model.to(device)
    model.train()
    learning_rate = (
        spec.learning_rate
        if run["initialization"] == "pretrained"
        else spec.random_init_learning_rate
    )
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=spec.weight_decay
    )
    scheduler = _scheduler(optimizer, spec)
    use_amp = device.type == "cuda"
    use_bf16 = use_amp and spec.bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not use_bf16)
    collator = AnswerOnlyCollator(max_seq_len=base_config.max_seq_len)
    rng = random.Random(adaptation_seed)
    order = list(range(spec.shots))
    cursor = len(order)
    accumulated = 0
    train_loss_sum = 0.0
    train_examples = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for step in range(spec.max_steps):
        if cursor + spec.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + spec.batch_size]
        cursor += spec.batch_size
        batch = collator([support_records[index] for index in indices])
        if batch["input_ids"].shape[0] * batch["input_ids"].shape[1] > spec.max_tokens_per_batch:
            raise ValueError("few-shot batch exceeds the frozen padded-token budget")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, use_amp, amp_dtype):
            logits = _model_logits(model, input_ids, attention_mask)
            per_example_loss = _causal_loss(logits, labels, reduction="none")
            loss = per_example_loss.mean()
        scaler.scale(loss / spec.gradient_accumulation_steps).backward()
        accumulated += 1
        train_loss_sum += float(per_example_loss.detach().sum())
        train_examples += len(indices)
        if accumulated == spec.gradient_accumulation_steps:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accumulated = 0
    if accumulated:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    validation_dataset = StreamingPermutationDataset(
        spec.validation_manifest,
        tasks=(str(run["task"]),),
        shuffle_buffer_size=1,
        seed=int(run["seed"]),
        rank=0,
        world_size=1,
    )
    validation = evaluate_records(
        model,
        validation_dataset,
        task_names=(str(run["task"]),),
        device=device,
        max_seq_len=base_config.max_seq_len,
        max_examples=8,
        max_padded_tokens=8192,
        amp_enabled=use_amp,
        amp_dtype=amp_dtype,
    )[str(run["task"])]
    if validation["examples"] != spec.expected_validation_examples:
        raise ValueError("few-shot validation did not consume the full task split")
    if not all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("few-shot model contains a non-finite tensor")
    checkpoint = {
        **identity,
        "model": model.state_dict(),
        "base_train_config": asdict(base_config),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "validation": validation,
        "train_mean_example_loss": train_loss_sum / train_examples,
        "train_examples": train_examples,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    marker = {
        **identity,
        "status": "completed",
        "checkpoint": _portable_path(checkpoint_path, spec.repository),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "validation": validation,
        "train_mean_example_loss": checkpoint["train_mean_example_loss"],
        "train_examples": train_examples,
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }
    _atomic_json(marker, marker_path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def _locked(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"controller lock already exists: {path}") from error


def run_all(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    spec = load_spec(config_path)
    implementation_commit = _git_commit(spec.repository)
    artifact = load_support_artifact(spec)
    support_sha256 = _sha256(spec.support_artifact)
    support = _support_lookup(artifact)
    audit = audit_experiment(spec.base_config_path, matrix="nested")
    if not audit["ok"]:
        raise ValueError("all 30 nested base models must pass strict audit")
    plan = build_plan(spec, audit["runs"])
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    lock_path = spec.output_dir / ".controller.lock"
    descriptor = _locked(lock_path)
    os.close(descriptor)
    try:
        markers = []
        for index, run in enumerate(plan, start=1):
            print(f"few-shot run {index}/{len(plan)}: {run['run_id']}", flush=True)
            markers.append(
                _train_one(
                    spec,
                    run,
                    support[(str(run["task"]), int(run["seed"]))],
                    support_sha256=support_sha256,
                    implementation_commit=implementation_commit,
                    device=device,
                )
            )
        summary = {
            "format_version": FORMAT_VERSION,
            "status": "completed",
            "implementation_commit": implementation_commit,
            "fewshot_config_sha256": spec.config_sha256,
            "support_artifact_sha256": support_sha256,
            "run_count": len(markers),
            "validation_examples": sum(
                int(marker["validation"]["examples"]) for marker in markers
            ),
            "runs": [
                {
                    "run_id": marker["run_id"],
                    "checkpoint_sha256": marker["checkpoint_sha256"],
                }
                for marker in markers
            ],
        }
        _atomic_json(summary, spec.output_dir / "manifest.json")
        return summary
    finally:
        lock_path.unlink(missing_ok=True)


def audit_all(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    spec = load_spec(config_path)
    artifact = load_support_artifact(spec)
    support_sha256 = _sha256(spec.support_artifact)
    support = _support_lookup(artifact)
    base_audit = audit_experiment(spec.base_config_path, matrix="nested")
    if not base_audit["ok"]:
        raise ValueError("base nested audit failed")
    plan = build_plan(spec, base_audit["runs"])
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
            if not isinstance(implementation_commit, str) or len(implementation_commit) != 40 or any(
                character not in "0123456789abcdef"
                for character in implementation_commit
            ):
                issues.append("invalid implementation commit")
            identity = _run_identity(
                spec,
                run,
                support_sha256=support_sha256,
                support_ids=[
                    int(record["id"])
                    for record in support[(str(run["task"]), int(run["seed"]))]
                ],
                implementation_commit=str(implementation_commit),
            )
            if not _completion_valid(marker_path, identity, checkpoint_path):
                issues.append("completion identity or checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                issues.append("checkpoint identity mismatch")
            config = TrainConfig.from_value(checkpoint["base_train_config"])
            if config.architecture != run["architecture"]:
                issues.append("base model architecture mismatch")
            model = _default_model_factory(config)
            model.load_state_dict(checkpoint["model"], strict=True)
            if not checkpoint["model"]:
                issues.append("empty model state")
            if not all(
                torch.isfinite(tensor).all().item()
                for tensor in checkpoint["model"].values()
            ):
                issues.append("non-finite model tensor")
            if checkpoint.get("validation") != marker.get("validation"):
                issues.append("marker/checkpoint validation mismatch")
            validation = marker.get("validation", {})
            if validation.get("examples") != spec.expected_validation_examples:
                issues.append("validation example count mismatch")
            if not isinstance(validation.get("tokens"), int) or validation.get("tokens", 0) < 1:
                issues.append("validation supervised-token count mismatch")
            for metric in ("loss", "token_accuracy", "sequence_accuracy"):
                value = validation.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    issues.append(f"non-finite validation {metric}")
                elif metric != "loss" and not 0 <= float(value) <= 1:
                    issues.append(f"out-of-range validation {metric}")
                elif metric == "loss" and float(value) < 0:
                    issues.append("negative validation loss")
            if marker.get("train_examples") != spec.max_steps * spec.batch_size:
                issues.append("training example count mismatch")
            if checkpoint.get("train_examples") != marker.get("train_examples"):
                issues.append("marker/checkpoint training count mismatch")
            train_loss = marker.get("train_mean_example_loss")
            if not isinstance(train_loss, (int, float)) or not math.isfinite(float(train_loss)) or float(train_loss) < 0:
                issues.append("invalid training loss")
            if checkpoint.get("train_mean_example_loss") != train_loss:
                issues.append("marker/checkpoint training loss mismatch")
            initial_lr = (
                spec.learning_rate
                if run["initialization"] == "pretrained"
                else spec.random_init_learning_rate
            )
            expected_final_lr = initial_lr * spec.min_learning_rate_ratio
            final_lr = checkpoint.get("final_learning_rate")
            if not isinstance(final_lr, float) or not math.isclose(
                final_lr, expected_final_lr, rel_tol=1e-9, abs_tol=1e-15
            ):
                issues.append("final learning rate mismatch")
            del model, checkpoint
        except Exception as error:  # audit must report malformed artifacts
            issues.append(f"{type(error).__name__}: {error}")
        if issues:
            failed.append({"run_id": run["run_id"], "issues": issues})
        else:
            passed.append(
                {
                    "run_id": run["run_id"],
                    "checkpoint": _portable_path(checkpoint_path, spec.repository),
                    "checkpoint_sha256": marker["checkpoint_sha256"],
                }
            )
    partials = sorted(
        str(path.relative_to(spec.repository))
        for path in spec.output_dir.rglob("*.tmp")
    ) if spec.output_dir.exists() else []
    return {
        "format_version": FORMAT_VERSION,
        "status": "passed" if len(passed) == spec.expected_runs and not failed else "incomplete" if not failed else "failed",
        "ok": len(passed) == spec.expected_runs and not failed and not partials,
        "expected_run_count": spec.expected_runs,
        "passed_count": len(passed),
        "incomplete_count": len(incomplete),
        "failed_count": len(failed),
        "partial_artifacts": partials,
        "passed": passed,
        "incomplete": incomplete,
        "failed": failed,
    }


def _test_identity(
    marker: Mapping[str, Any],
    *,
    test_manifest_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": TEST_FORMAT_VERSION,
        "run_id": marker["run_id"],
        "initialization": marker["initialization"],
        "architecture": marker["architecture"],
        "base_run_id": marker["base_run_id"],
        "base_trained_task_count": marker["base_trained_task_count"],
        "seed": marker["seed"],
        "task": marker["task"],
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "fewshot_config_sha256": marker["fewshot_config_sha256"],
        "support_artifact_sha256": marker["support_artifact_sha256"],
        "test_manifest_sha256": test_manifest_sha256,
        "evaluator_commit": evaluator_commit,
    }


def evaluate_test(
    config_path: Path = DEFAULT_CONFIG,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    spec = load_spec(config_path)
    evaluator_commit = _git_commit(spec.repository)
    audit = audit_all(config_path)
    if not audit["ok"]:
        raise ValueError("all 144 few-shot runs must pass strict audit before test")
    verification = verify_manifest(spec.test_manifest, full=True, workers=1)
    if not verification["ok"] or verification["record_count"] != 100_000:
        raise ValueError("few-shot test manifest failed full verification")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    lock_path = spec.evaluation_dir / ".evaluation.lock"
    descriptor = _locked(lock_path)
    os.close(descriptor)
    results: list[dict[str, Any]] = []
    try:
        for index, passed in enumerate(audit["passed"], start=1):
            checkpoint_path = spec.repository / passed["checkpoint"]
            marker_path = checkpoint_path.parent / "completed.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            identity = _test_identity(
                marker,
                test_manifest_sha256=spec.test_manifest_sha256,
                evaluator_commit=evaluator_commit,
            )
            output_path = spec.evaluation_dir / "per-run" / f"{marker['run_id']}.json"
            if output_path.is_file():
                result = json.loads(output_path.read_text(encoding="utf-8"))
                if any(result.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"test result identity mismatch: {output_path}")
                if result.get("status") != "completed" or result.get("metrics", {}).get("examples") != spec.expected_test_examples:
                    raise ValueError(f"test result is incomplete: {output_path}")
                results.append(result)
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            config = TrainConfig.from_value(checkpoint["base_train_config"])
            model = _default_model_factory(config)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            use_amp = device.type == "cuda"
            amp_dtype = (
                torch.bfloat16
                if use_amp and spec.bf16 and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            dataset = StreamingPermutationDataset(
                spec.test_manifest,
                tasks=(str(marker["task"]),),
                shuffle_buffer_size=1,
                seed=int(marker["seed"]),
                rank=0,
                world_size=1,
            )
            print(f"few-shot test {index}/{spec.expected_runs}: {marker['run_id']}", flush=True)
            started = time.monotonic()
            metrics = evaluate_records(
                model,
                dataset,
                task_names=(str(marker["task"]),),
                device=device,
                max_seq_len=config.max_seq_len,
                max_examples=8,
                max_padded_tokens=8192,
                amp_enabled=use_amp,
                amp_dtype=amp_dtype,
            )[str(marker["task"])]
            if metrics["examples"] != spec.expected_test_examples:
                raise ValueError("few-shot test did not consume the full task split")
            result = {
                **identity,
                "status": "completed",
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            _atomic_json(result, output_path)
            results.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        manifest = {
            "format_version": TEST_FORMAT_VERSION,
            "status": "completed",
            "evaluator_commit": evaluator_commit,
            "fewshot_config_sha256": spec.config_sha256,
            "support_artifact_sha256": _sha256(spec.support_artifact),
            "test_manifest_sha256": spec.test_manifest_sha256,
            "test_manifest_full_verification": verification,
            "run_count": len(results),
            "examples_per_run": spec.expected_test_examples,
            "total_model_examples": len(results) * spec.expected_test_examples,
            "runs": [
                {
                    "run_id": result["run_id"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "result_file": f"per-run/{result['run_id']}.json",
                }
                for result in results
            ],
        }
        if len(results) != spec.expected_runs:
            raise ValueError("few-shot test result count is incomplete")
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
        output: Mapping[str, Any] = {
            "status": "completed",
            "support_artifact": str(spec.support_artifact),
            "record_count": artifact["record_count"],
        }
    elif args.command == "plan":
        audit = audit_experiment(spec.base_config_path, matrix="nested")
        plan = build_plan(spec, audit["runs"])
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

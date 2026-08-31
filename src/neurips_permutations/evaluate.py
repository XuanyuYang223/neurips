"""One-time, resumable evaluation of audited models on test shard 099."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import tomllib
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .audit import audit_experiment
from .experiments import task_names_for_experiment
from .training import (
    AnswerOnlyCollator,
    StreamingPermutationDataset,
    TrainConfig,
    _autocast_context,
    _default_model_factory,
    _model_logits,
    resolve_shards,
)
from .verify import verify_manifest


DEFAULT_CONFIG = Path("configs/henry_permutation_revised.toml")
DEFAULT_OUTPUT_DIR = Path("results/v3/evaluation")
EVALUATION_FORMAT_VERSION = "v3-test-evaluation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repository: Path) -> str:
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
        if completed.returncode != 0:
            raise RuntimeError("tracked worktree must be clean before test evaluation")
    return commit


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def task_homogeneous_batches(
    records: Iterable[Mapping[str, Any]],
    *,
    task_names: Sequence[str],
    max_examples: int,
    max_padded_tokens: int,
) -> Iterator[tuple[str, list[Mapping[str, Any]]]]:
    """Stream once while batching each task separately to reduce padding."""

    if max_examples < 1 or max_padded_tokens < 1:
        raise ValueError("batch limits must be positive")
    allowed = set(task_names)
    pending: dict[str, list[Mapping[str, Any]]] = {task: [] for task in task_names}
    longest = {task: 0 for task in task_names}
    for record in records:
        task = record.get("task")
        if task not in allowed:
            raise ValueError(f"unexpected task in test data: {task!r}")
        tokens = record.get("tokens")
        if not isinstance(tokens, (list, tuple)):
            raise ValueError("test record has invalid tokens")
        length = len(tokens)
        if length > max_padded_tokens:
            raise ValueError(
                f"sequence length {length} exceeds test padded-token budget"
            )
        batch = pending[task]
        proposed_longest = max(longest[task], length)
        if batch and (
            len(batch) >= max_examples
            or proposed_longest * (len(batch) + 1) > max_padded_tokens
        ):
            yield task, batch
            batch = []
            pending[task] = batch
            longest[task] = 0
        batch.append(record)
        longest[task] = max(longest[task], length)
    for task in task_names:
        if pending[task]:
            yield task, pending[task]


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    correct_tokens: int = 0
    supervised_tokens: int = 0
    correct_sequences: int = 0
    examples: int = 0

    def result(self) -> dict[str, float | int]:
        if self.examples < 1 or self.supervised_tokens < 1:
            raise ValueError("cannot finalize an empty task metric")
        values: dict[str, float | int] = {
            "loss": self.loss_sum / self.supervised_tokens,
            "token_accuracy": self.correct_tokens / self.supervised_tokens,
            "sequence_accuracy": self.correct_sequences / self.examples,
            "tokens": self.supervised_tokens,
            "examples": self.examples,
        }
        if not all(
            math.isfinite(float(values[key]))
            for key in ("loss", "token_accuracy", "sequence_accuracy")
        ):
            raise ValueError("test metric is non-finite")
        return values


@torch.inference_mode()
def evaluate_records(
    model: nn.Module,
    records: Iterable[Mapping[str, Any]],
    *,
    task_names: Sequence[str],
    device: torch.device,
    max_seq_len: int,
    max_examples: int = 32,
    max_padded_tokens: int = 8_192,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    progress_every: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Evaluate all records once and return token/exact metrics per task."""

    model.eval()
    collator = AnswerOnlyCollator(max_seq_len=max_seq_len)
    accumulators = {task: MetricAccumulator() for task in task_names}
    processed = 0
    next_progress = progress_every
    for task, raw_batch in task_homogeneous_batches(
        records,
        task_names=task_names,
        max_examples=max_examples,
        max_padded_tokens=max_padded_tokens,
    ):
        batch = collator(raw_batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits = _model_logits(model, input_ids, attention_mask)
            token_losses = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape[0], -1)
        targets = labels[:, 1:]
        predictions = logits[:, :-1, :].argmax(dim=-1)
        supervised = targets.ne(-100)
        matches = predictions.eq(targets) | ~supervised
        counts = supervised.sum(dim=1)
        accumulator = accumulators[task]
        accumulator.loss_sum += float((token_losses * supervised).sum())
        accumulator.correct_tokens += int(
            (predictions.eq(targets) & supervised).sum()
        )
        accumulator.supervised_tokens += int(supervised.sum())
        accumulator.correct_sequences += int(
            (matches.all(dim=1) & counts.gt(0)).sum()
        )
        accumulator.examples += input_ids.shape[0]
        processed += input_ids.shape[0]
        if next_progress is not None and processed >= next_progress:
            print(f"test examples processed: {processed}", flush=True)
            next_progress += progress_every or 0
    return {task: accumulators[task].result() for task in task_names}


def _expected_result_identity(
    *,
    matrix: str,
    run: Mapping[str, Any],
    config_sha256: str,
    test_manifest_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": EVALUATION_FORMAT_VERSION,
        "matrix": matrix,
        "condition": run.get("condition", ""),
        "run_id": run["run_id"],
        "architecture": run["architecture"],
        "trained_tasks": run["tasks"],
        "trained_task_count": run["task_count"],
        "seed": run["seed"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "experiment_config_sha256": config_sha256,
        "test_manifest_sha256": test_manifest_sha256,
        "test_shards": "099",
        "evaluator_commit": evaluator_commit,
    }


def _load_completed_result(
    path: Path,
    identity: Mapping[str, Any],
    task_names: Sequence[str],
    *,
    expected_examples_per_task: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume invalid result {path}: {error}") from error
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in identity.items()):
        raise ValueError(f"existing result identity differs: {path}")
    metrics = payload.get("metrics")
    if payload.get("status") != "completed" or not isinstance(metrics, dict):
        raise ValueError(f"existing result is incomplete: {path}")
    if set(metrics) != set(task_names):
        raise ValueError(f"existing result task grid differs: {path}")
    if any(metric.get("examples") != expected_examples_per_task for metric in metrics.values()):
        raise ValueError(f"existing result example counts differ: {path}")
    return payload


def _evaluate_one_run(
    *,
    matrix: str,
    run: Mapping[str, Any],
    test_shards: Sequence[Path],
    task_names: Sequence[str],
    identity: Mapping[str, Any],
    output_path: Path,
    device: torch.device,
    expected_examples_per_task: int,
    max_examples: int,
    max_padded_tokens: int,
) -> dict[str, Any]:
    resumed = _load_completed_result(
        output_path,
        identity,
        task_names,
        expected_examples_per_task=expected_examples_per_task,
    )
    if resumed is not None:
        print(f"test result already complete: {run['run_id']}", flush=True)
        return resumed

    checkpoint_path = Path(run["checkpoint_path"])
    if _sha256(checkpoint_path) != run["checkpoint_sha256"]:
        raise ValueError(f"checkpoint changed after strict audit: {run['run_id']}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_config = TrainConfig.from_value(checkpoint["config"])
    model = _default_model_factory(train_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device)
    use_amp = train_config.amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and train_config.bf16 and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    dataset = StreamingPermutationDataset(
        test_shards,
        shuffle_buffer_size=1,
        seed=train_config.seed,
        rank=0,
        world_size=1,
    )
    started = time.monotonic()
    print(f"evaluating test shard: {run['run_id']}", flush=True)
    metrics = evaluate_records(
        model,
        dataset,
        task_names=task_names,
        device=device,
        max_seq_len=train_config.max_seq_len,
        max_examples=max_examples,
        max_padded_tokens=max_padded_tokens,
        amp_enabled=use_amp,
        amp_dtype=amp_dtype,
        progress_every=50_000,
    )
    for task, metric in metrics.items():
        if metric["examples"] != expected_examples_per_task:
            raise ValueError(
                f"{run['run_id']}:{task} evaluated {metric['examples']} examples, "
                f"expected {expected_examples_per_task}"
            )
    payload = {
        **identity,
        "status": "completed",
        "device_type": device.type,
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if use_amp else "float32",
        "max_examples_per_batch": max_examples,
        "max_padded_tokens": max_padded_tokens,
        "examples": sum(int(metric["examples"]) for metric in metrics.values()),
        "elapsed_seconds": time.monotonic() - started,
        "metrics": metrics,
    }
    _atomic_json(payload, output_path)
    print(
        f"completed test evaluation: {run['run_id']} "
        f"({payload['elapsed_seconds']:.2f}s)",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def evaluate_all(
    config_path: Path,
    output_dir: Path,
    *,
    max_examples: int = 32,
    max_padded_tokens: int = 8_192,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Strictly gate and evaluate all 48 frozen models exactly once."""

    repository = config_path.resolve().parent.parent
    evaluator_commit = _git_commit(repository)
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    output_absolute = (repository / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    run_root = (repository / config["output_dir"]).resolve()
    if output_absolute == run_root or run_root in output_absolute.parents:
        raise ValueError("evaluation outputs must be outside the training run tree")
    output_absolute.mkdir(parents=True, exist_ok=True)
    lock_path = output_absolute / ".evaluation.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"evaluation controller lock already exists: {lock_path}") from error
    os.close(descriptor)
    try:
        audits = {
            matrix: audit_experiment(config_path, matrix=matrix)
            for matrix in ("nested", "category")
        }
        if not all(audit["ok"] for audit in audits.values()):
            raise ValueError("all 48 training runs must pass strict audit")
        test_manifest = repository / config["test_manifest"]
        test_manifest_sha256 = _sha256(test_manifest)
        declared_test_sha256 = config["dataset_artifact"]["test_manifest_sha256"]
        if test_manifest_sha256 != declared_test_sha256:
            raise ValueError("test split manifest differs from the frozen config")
        verification = verify_manifest(test_manifest, full=True, workers=1)
        if not verification["ok"] or verification["record_count"] != 100_000:
            raise ValueError("test split failed full verification")
        task_names = tuple(task_names_for_experiment(config))
        expected_examples_per_task = verification["record_count"] // len(task_names)
        test_shards = resolve_shards(test_manifest)
        device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        per_run_dir = output_absolute / "per-run"
        results: list[dict[str, Any]] = []
        for matrix in ("nested", "category"):
            for audited_run in audits[matrix]["runs"]:
                run = dict(audited_run)
                checkpoint_path = Path(run["checkpoint_path"])
                if not checkpoint_path.is_absolute():
                    run["checkpoint_path"] = str(repository / checkpoint_path)
                identity = _expected_result_identity(
                    matrix=matrix,
                    run=run,
                    config_sha256=config_sha256,
                    test_manifest_sha256=test_manifest_sha256,
                    evaluator_commit=evaluator_commit,
                )
                results.append(
                    _evaluate_one_run(
                        matrix=matrix,
                        run=run,
                        test_shards=test_shards,
                        task_names=task_names,
                        identity=identity,
                        output_path=per_run_dir / f"{run['run_id']}.json",
                        device=device,
                        expected_examples_per_task=expected_examples_per_task,
                        max_examples=max_examples,
                        max_padded_tokens=max_padded_tokens,
                    )
                )
        summary = {
            "format_version": EVALUATION_FORMAT_VERSION,
            "status": "completed",
            "evaluator_commit": evaluator_commit,
            "experiment_config_sha256": config_sha256,
            "test_manifest": str(test_manifest.relative_to(repository)),
            "test_manifest_sha256": test_manifest_sha256,
            "test_shards": "099",
            "test_manifest_full_verification": verification,
            "run_count": len(results),
            "task_count": len(task_names),
            "examples_per_run": verification["record_count"],
            "examples_per_task_per_run": expected_examples_per_task,
            "total_model_examples": len(results) * verification["record_count"],
            "runs": [
                {
                    "matrix": result["matrix"],
                    "run_id": result["run_id"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "result_file": f"per-run/{result['run_id']}.json",
                }
                for result in results
            ],
        }
        if len(results) != 48:
            raise ValueError(f"expected 48 test results, found {len(results)}")
        _atomic_json(summary, output_absolute / "manifest.json")
        return summary
    finally:
        lock_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--max-padded-tokens", type=int, default=8_192)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_all(
        args.config,
        args.output_dir,
        max_examples=args.max_examples,
        max_padded_tokens=args.max_padded_tokens,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_count": summary["run_count"],
                "total_model_examples": summary["total_model_examples"],
                "output": str(args.output_dir / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

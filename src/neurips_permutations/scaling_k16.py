"""Run, audit, evaluate, and report the deadline-scoped k=16 scaling study."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import tomllib
from typing import Any

import torch

from .audit import audit_experiment
from .cka import _atomic_csv, _atomic_json, _atomic_text, _git_commit, _sha256
from .evaluate import (
    _evaluate_one_run,
    _expected_result_identity,
    _sha256 as _evaluation_sha256,
)
from .experiments import build_matrix, run_matrix, task_names_for_experiment
from .training import resolve_shards
from .verify import verify_manifest


DEFAULT_CONFIG = Path("configs/permutation_scaling_k16.toml")
CONDITION_ORDER = (
    "baseline",
    "data10x_model1x",
    "data1x_model2x",
    "data10x_model2x",
)
ARCHITECTURES = ("transformer", "mlp")
MODEL_SEEDS = (17, 42, 314159)
NEW_SEEDS = (42, 314159)
TASK_COUNT = 16
EVALUATION_FORMAT = "permutation-scaling-k16-test/v1"
RESULT_FORMAT = "permutation-scaling-k16-results/v1"


def _load(path: Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any], str]:
    absolute = path.resolve()
    repository = absolute.parent.parent
    payload = absolute.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if config.get("protocol_version") != "permutation-scaling-k16/v1":
        raise ValueError("unexpected k16 scaling protocol")
    if config.get("task_count") != TASK_COUNT:
        raise ValueError("k16 scaling task count drifted")
    if tuple(config.get("architectures", ())) != ARCHITECTURES:
        raise ValueError("k16 scaling architectures drifted")
    if tuple(config.get("model_seeds", ())) != MODEL_SEEDS:
        raise ValueError("k16 scaling seeds drifted")
    if tuple(config.get("new_training_seeds", ())) != NEW_SEEDS:
        raise ValueError("k16 scaling new seeds drifted")
    conditions = config.get("conditions")
    if not isinstance(conditions, dict) or tuple(conditions) != CONDITION_ORDER:
        raise ValueError("k16 scaling condition order drifted")
    for name, value in conditions.items():
        condition_path = repository / str(value["config"])
        if _sha256(condition_path) != value["config_sha256"]:
            raise ValueError(f"frozen condition config changed: {name}")
    test_manifest = repository / str(config["test_manifest"])
    if _sha256(test_manifest) != config["test_manifest_sha256"]:
        raise ValueError("frozen scaling test manifest changed")
    if tuple(config.get("primary_tasks", ())) != (
        "to_reduced_word",
        "compose",
        "to_lehmer",
    ) or config.get("diagnostic_task") != "parity":
        raise ValueError("k16 scaling outcome tasks drifted")
    return repository, config, digest


def _selected_runs(condition_path: Path) -> list[Any]:
    selected = [run for run in build_matrix(condition_path) if run.task_count == TASK_COUNT]
    if len(selected) != 6 or {
        (run.architecture, run.seed) for run in selected
    } != {(architecture, seed) for architecture in ARCHITECTURES for seed in MODEL_SEEDS}:
        raise ValueError("condition does not contain the frozen six k16 runs")
    return selected


def plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    repository, config, digest = _load(config_path)
    rows = []
    for condition in CONDITION_ORDER:
        value = config["conditions"][condition]
        condition_path = repository / value["config"]
        for run in _selected_runs(condition_path):
            rows.append(
                {
                    "condition": condition,
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "run_id": run.run_id,
                    "output_dir": run.output_dir,
                    "train_new": bool(value["train_new"] and run.seed in NEW_SEEDS),
                    "completed": (Path(run.output_dir) / "completed.json").is_file(),
                }
            )
    if len(rows) != 24:
        raise ValueError("k16 scaling plan must contain 24 endpoints")
    return {
        "protocol_sha256": digest,
        "run_count": 24,
        "new_run_count": sum(bool(row["train_new"]) for row in rows),
        "completed_count": sum(bool(row["completed"]) for row in rows),
        "runs": rows,
    }


def train_missing(
    config_path: Path = DEFAULT_CONFIG,
    *,
    architecture: str | None = None,
) -> int:
    repository, config, _ = _load(config_path)
    if architecture is not None and architecture not in ARCHITECTURES:
        raise ValueError("architecture must be transformer or mlp")
    for condition in CONDITION_ORDER[1:]:
        condition_path = repository / config["conditions"][condition]["config"]
        selected = [
            run.run_id
            for run in _selected_runs(condition_path)
            if run.seed in NEW_SEEDS
            and (architecture is None or run.architecture == architecture)
        ]
        result = run_matrix(condition_path, matrix="nested", only=selected)
        if result:
            return result
    return 0


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    repository, config, digest = _load(config_path)
    conditions: dict[str, Any] = {}
    all_runs = []
    for condition in CONDITION_ORDER:
        condition_path = repository / config["conditions"][condition]["config"]
        report = audit_experiment(condition_path, matrix="nested")
        selected = [run for run in report["runs"] if run["task_count"] == TASK_COUNT]
        if len(selected) != 6:
            raise ValueError("audit did not return six k16 runs")
        conditions[condition] = {
            "config": str(config["conditions"][condition]["config"]),
            "config_sha256": report["config_sha256"],
            "manifest_statuses": {
                name: value["status"] for name, value in report["manifests"].items()
            },
            "issues": report["issues"],
            "passed_count": sum(run["status"] == "passed" for run in selected),
            "incomplete_count": sum(run["status"] == "incomplete" for run in selected),
            "failed_count": sum(run["status"] == "failed" for run in selected),
        }
        for run in selected:
            all_runs.append({"condition": condition, **run})
    ok = (
        len(all_runs) == 24
        and all(run["status"] == "passed" for run in all_runs)
        and all(not value["issues"] for value in conditions.values())
        and all(
            status == "passed"
            for value in conditions.values()
            for status in value["manifest_statuses"].values()
        )
    )
    return {
        "format_version": "permutation-scaling-k16-audit/v1",
        "status": "passed" if ok else "incomplete_or_failed",
        "ok": ok,
        "protocol_sha256": digest,
        "expected_run_count": 24,
        "passed_count": sum(run["status"] == "passed" for run in all_runs),
        "incomplete_count": sum(run["status"] == "incomplete" for run in all_runs),
        "failed_count": sum(run["status"] == "failed" for run in all_runs),
        "conditions": conditions,
        "runs": all_runs,
    }


def evaluate(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    repository, config, digest = _load(config_path)
    commit = _git_commit(repository)
    audit_result = audit(config_path)
    if not audit_result["ok"]:
        raise ValueError("all 24 k16 scaling endpoints must pass strict audit")
    test_manifest = repository / config["test_manifest"]
    verification = verify_manifest(test_manifest, full=True, workers=1)
    if not verification["ok"] or verification["record_count"] != 100_000:
        raise ValueError("frozen v3 test split failed full verification")
    tasks = tuple(task_names_for_experiment(
        tomllib.loads((repository / config["conditions"]["baseline"]["config"]).read_text())
    ))
    shards = resolve_shards(test_manifest)
    output = repository / config["evaluation_dir"]
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".evaluation.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"scaling evaluation lock exists: {lock}") from error
    os.close(descriptor)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    try:
        for audited in audit_result["runs"]:
            run = dict(audited)
            condition = str(run.pop("condition"))
            run["condition"] = condition
            checkpoint = Path(run["checkpoint_path"])
            if not checkpoint.is_absolute():
                run["checkpoint_path"] = str(repository / checkpoint)
            condition_config = config["conditions"][condition]
            identity = _expected_result_identity(
                matrix="scaling_k16",
                run=run,
                config_sha256=condition_config["config_sha256"],
                test_manifest_sha256=config["test_manifest_sha256"],
                evaluator_commit=commit,
            )
            result = _evaluate_one_run(
                matrix="scaling_k16",
                run=run,
                test_shards=shards,
                task_names=tasks,
                identity=identity,
                output_path=output / "per-run" / f"{condition}--{run['run_id']}.json",
                device=device,
                expected_examples_per_task=5_000,
                max_examples=32,
                max_padded_tokens=8_192,
            )
            results.append(result)
    finally:
        lock.unlink(missing_ok=True)
    manifest = {
        "format_version": EVALUATION_FORMAT,
        "status": "completed",
        "protocol_sha256": digest,
        "evaluator_commit": commit,
        "test_manifest_sha256": config["test_manifest_sha256"],
        "full_verification": verification,
        "run_count": len(results),
        "examples_per_run": 100_000,
        "examples_per_task": 5_000,
        "total_model_examples": len(results) * 100_000,
        "runs": [
            {
                "condition": result["condition"],
                "run_id": result["run_id"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_file": f"per-run/{result['condition']}--{result['run_id']}.json",
            }
            for result in results
        ],
    }
    if len(results) != 24:
        raise ValueError("k16 scaling evaluation must contain 24 runs")
    _atomic_json(manifest, output / "manifest.json")
    return manifest


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def export_results(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    repository, config, digest = _load(config_path)
    audit_result = audit(config_path)
    if not audit_result["ok"]:
        raise ValueError("scaling checkpoints failed strict audit")
    evaluation_dir = repository / config["evaluation_dir"]
    evaluation = json.loads((evaluation_dir / "manifest.json").read_text())
    if evaluation.get("format_version") != EVALUATION_FORMAT or evaluation.get("run_count") != 24:
        raise ValueError("scaling evaluation manifest is incomplete")
    primary = tuple(config["primary_tasks"])
    diagnostic = str(config["diagnostic_task"])
    rows = []
    for entry in evaluation["runs"]:
        payload = json.loads((evaluation_dir / entry["result_file"]).read_text())
        metrics = payload["metrics"]
        structured = [metrics[task] for task in primary]
        rows.append(
            {
                "condition": entry["condition"],
                "data_multiplier": config["conditions"][entry["condition"]]["data_multiplier"],
                "model_multiplier": config["conditions"][entry["condition"]]["model_multiplier"],
                "architecture": payload["architecture"],
                "seed": payload["seed"],
                "run_id": payload["run_id"],
                "structured_holdout_loss": statistics.fmean(float(x["loss"]) for x in structured),
                "structured_holdout_token_accuracy": statistics.fmean(float(x["token_accuracy"]) for x in structured),
                "structured_holdout_sequence_accuracy": statistics.fmean(float(x["sequence_accuracy"]) for x in structured),
                "parity_loss": metrics[diagnostic]["loss"],
                "parity_token_accuracy": metrics[diagnostic]["token_accuracy"],
                "parity_sequence_accuracy": metrics[diagnostic]["sequence_accuracy"],
                "examples_per_task": metrics[diagnostic]["examples"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
    if len(rows) != 24:
        raise ValueError("scaling raw result grid is incomplete")
    metrics = (
        "structured_holdout_loss",
        "structured_holdout_token_accuracy",
        "structured_holdout_sequence_accuracy",
        "parity_loss",
        "parity_token_accuracy",
        "parity_sequence_accuracy",
    )
    summary = []
    for condition in CONDITION_ORDER:
        for architecture in ARCHITECTURES:
            selected = [row for row in rows if row["condition"] == condition and row["architecture"] == architecture]
            if len(selected) != 3 or {row["seed"] for row in selected} != set(MODEL_SEEDS):
                raise ValueError("scaling summary cell lacks three seeds")
            record: dict[str, Any] = {
                "condition": condition,
                "data_multiplier": config["conditions"][condition]["data_multiplier"],
                "model_multiplier": config["conditions"][condition]["model_multiplier"],
                "architecture": architecture,
                "seed_count": 3,
            }
            for metric in metrics:
                mean, sd = _mean_sd([float(row[metric]) for row in selected])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sample_sd"] = sd
            summary.append(record)
    effects = []
    lookup = {(row["condition"], row["architecture"], row["seed"]): row for row in rows}
    contrasts = {
        "data_effect_at_1x_model": ("data10x_model1x", "baseline", 1, -1),
        "model_effect_at_1x_data": ("data1x_model2x", "baseline", 1, -1),
        "data_effect_at_2x_model": ("data10x_model2x", "data1x_model2x", 1, -1),
        "model_effect_at_10x_data": ("data10x_model2x", "data10x_model1x", 1, -1),
    }
    for architecture in ARCHITECTURES:
        for name, (left, right, left_sign, right_sign) in contrasts.items():
            values = [
                left_sign * float(lookup[left, architecture, seed]["structured_holdout_sequence_accuracy"])
                + right_sign * float(lookup[right, architecture, seed]["structured_holdout_sequence_accuracy"])
                for seed in MODEL_SEEDS
            ]
            mean, sd = _mean_sd(values)
            effects.append({"architecture": architecture, "contrast": name, "mean": mean, "sample_sd": sd})
        interaction = [
            float(lookup["data10x_model2x", architecture, seed]["structured_holdout_sequence_accuracy"])
            - float(lookup["data10x_model1x", architecture, seed]["structured_holdout_sequence_accuracy"])
            - float(lookup["data1x_model2x", architecture, seed]["structured_holdout_sequence_accuracy"])
            + float(lookup["baseline", architecture, seed]["structured_holdout_sequence_accuracy"])
            for seed in MODEL_SEEDS
        ]
        mean, sd = _mean_sd(interaction)
        effects.append({"architecture": architecture, "contrast": "data_by_model_interaction", "mean": mean, "sample_sd": sd})
    output = repository / config["results_dir"]
    raw_fields = tuple(rows[0])
    summary_fields = tuple(summary[0])
    effect_fields = ("architecture", "contrast", "mean", "sample_sd")
    _atomic_csv(rows, raw_fields, output / "model_results.csv")
    _atomic_csv(summary, summary_fields, output / "summary.csv")
    _atomic_csv(effects, effect_fields, output / "factorial_effects.csv")
    lines = [
        "# k=16 data and model scaling results",
        "",
        "The primary metric is exact sequence accuracy macro-averaged over `to_reduced_word`, `compose`, and `to_lehmer`. Values are mean +/- sample SD over three paired model seeds.",
        "",
        "| Architecture | Condition | Structured exact | Parity exact |",
        "|---|---|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['architecture']} | `{row['condition']}` | "
            f"{100 * row['structured_holdout_sequence_accuracy_mean']:.3f}% +/- {100 * row['structured_holdout_sequence_accuracy_sample_sd']:.3f}% | "
            f"{100 * row['parity_sequence_accuracy_mean']:.3f}% +/- {100 * row['parity_sequence_accuracy_sample_sd']:.3f}% |"
        )
    lines.extend(["", "The table is descriptive with only three seeds. See `factorial_effects.csv` for paired data, depth, and interaction contrasts and `model_results.csv` for every endpoint.", ""])
    _atomic_text("\n".join(lines), output / "README.md")
    manifest = {
        "format_version": RESULT_FORMAT,
        "status": "completed",
        "protocol_sha256": digest,
        "evaluation_manifest_sha256": _sha256(evaluation_dir / "manifest.json"),
        "row_count": len(rows),
        "summary_row_count": len(summary),
        "artifacts": {
            name: _sha256(output / name)
            for name in ("model_results.csv", "summary.csv", "factorial_effects.csv", "README.md")
        },
    }
    if not all(math.isfinite(float(value)) for row in rows for key, value in row.items() if key.endswith(("loss", "accuracy"))):
        raise ValueError("non-finite scaling result")
    _atomic_json(manifest, output / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "audit", "evaluate", "results"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "plan":
        value = plan(args.config)
    elif args.action == "run":
        return train_missing(args.config, architecture=args.architecture)
    elif args.action == "audit":
        value = audit(args.config)
    elif args.action == "evaluate":
        value = evaluate(args.config, device_name=args.device)
    else:
        value = export_results(args.config)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0 if value.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

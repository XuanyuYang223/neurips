"""Evaluate the v3 nested models on permutations of length 31 through 40."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .audit import audit_experiment
from .cka import _atomic_csv, _atomic_text
from .evaluate import _atomic_json, _git_commit, _sha256, evaluate_records
from .generate import V3_TASK_NAMES
from .training import StreamingPermutationDataset, TrainConfig, _default_model_factory
from .verify import verify_manifest


DEFAULT_CONFIG = Path("configs/v3_size_extrapolation.toml")
FIXED_TRAIN_HOLDOUTS = frozenset(
    {"to_reduced_word", "compose", "parity", "to_lehmer"}
)
RAW_FIELDS = (
    "run_id",
    "architecture",
    "trained_task_count",
    "seed",
    "task",
    "task_status",
    "examples",
    "supervised_tokens",
    "loss",
    "token_accuracy",
    "sequence_accuracy",
    "in_domain_loss",
    "in_domain_token_accuracy",
    "in_domain_sequence_accuracy",
    "loss_delta_out_minus_in",
    "token_accuracy_delta_out_minus_in",
    "sequence_accuracy_delta_out_minus_in",
)
SUMMARY_FIELDS = (
    "architecture",
    "trained_task_count",
    "task_status",
    "task_count",
    "seed_count",
    "loss_mean",
    "loss_sample_sd",
    "token_accuracy_mean",
    "token_accuracy_sample_sd",
    "sequence_accuracy_mean",
    "sequence_accuracy_sample_sd",
    "in_domain_sequence_accuracy_mean",
    "sequence_accuracy_delta_out_minus_in_mean",
    "sequence_accuracy_delta_out_minus_in_sample_sd",
)


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    config_path = path.resolve()
    repository = config_path.parent.parent
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != "v3-size-extrapolation/v1":
        raise ValueError("unsupported size-extrapolation protocol")
    for key, hash_key in (
        ("base_experiment_config", "base_experiment_config_sha256"),
        ("data_manifest", "data_manifest_sha256"),
    ):
        artifact = repository / str(config[key])
        if _sha256(artifact) != config[hash_key]:
            raise ValueError(f"{key} differs from the frozen protocol")
    if (config.get("min_entries"), config.get("max_entries")) != (31, 40):
        raise ValueError("size extrapolation must use n=31 through n=40")
    if config.get("excluded_tasks") != ["to_reduced_word"]:
        raise ValueError("the frozen size-extrapolation exclusion differs")
    return config, repository


def evaluation_tasks() -> tuple[str, ...]:
    return tuple(task for task in V3_TASK_NAMES if task != "to_reduced_word")


def _identity(
    run: Mapping[str, Any],
    *,
    manifest_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    return {
        "format_version": "v3-size-extrapolation-evaluation/v1",
        "run_id": run["run_id"],
        "architecture": run["architecture"],
        "trained_tasks": run["tasks"],
        "trained_task_count": run["task_count"],
        "seed": run["seed"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "data_manifest_sha256": manifest_sha256,
        "evaluation_tasks": list(evaluation_tasks()),
        "min_entries": 31,
        "max_entries": 40,
        "evaluator_commit": evaluator_commit,
    }


def evaluate_all(
    config_path: Path = DEFAULT_CONFIG,
    *,
    device_name: str | None = None,
) -> dict[str, Any]:
    config, repository = _load_config(config_path)
    evaluator_commit = _git_commit(repository)
    base_config = repository / str(config["base_experiment_config"])
    audit = audit_experiment(base_config, matrix="nested")
    if not audit["ok"] or audit["passed_count"] != int(config["model_count"]):
        raise ValueError("all 30 v3 nested models must pass strict audit")
    data_manifest = repository / str(config["data_manifest"])
    verification = verify_manifest(data_manifest, full=True, workers=1)
    if (
        not verification["ok"]
        or verification["record_count"] != int(config["records"])
        or verification["task_counts"] != {task: 1000 for task in V3_TASK_NAMES}
    ):
        raise ValueError("size-extrapolation corpus failed full verification")
    tasks = evaluation_tasks()
    output_dir = repository / str(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".evaluation.lock"
    try:
        descriptor = lock.open("x")
    except FileExistsError as error:
        raise RuntimeError("size-extrapolation controller is already active") from error
    descriptor.close()
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = []
    try:
        for index, run_value in enumerate(audit["runs"], start=1):
            run = dict(run_value)
            checkpoint_path = Path(str(run["checkpoint_path"]))
            if not checkpoint_path.is_absolute():
                checkpoint_path = repository / checkpoint_path
            identity = _identity(
                run,
                manifest_sha256=str(config["data_manifest_sha256"]),
                evaluator_commit=evaluator_commit,
            )
            output = output_dir / "per-run" / f"{run['run_id']}.json"
            if output.is_file():
                result = json.loads(output.read_text(encoding="utf-8"))
                if any(result.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"existing extrapolation result differs: {output}")
                if result.get("status") != "completed":
                    raise ValueError(f"existing extrapolation result is incomplete: {output}")
                results.append(result)
                continue
            if _sha256(checkpoint_path) != run["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed after audit: {run['run_id']}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            train_config = TrainConfig.from_value(checkpoint["config"])
            model = _default_model_factory(train_config)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            dataset = StreamingPermutationDataset(
                data_manifest,
                tasks=tasks,
                shuffle_buffer_size=1,
                seed=int(run["seed"]),
            )
            use_amp = train_config.amp and device.type == "cuda"
            amp_dtype = (
                torch.bfloat16
                if use_amp and train_config.bf16 and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            print(f"size extrapolation {index}/{len(audit['runs'])}: {run['run_id']}", flush=True)
            started = time.monotonic()
            metrics = evaluate_records(
                model,
                dataset,
                task_names=tasks,
                device=device,
                max_seq_len=train_config.max_seq_len,
                max_examples=int(config["max_examples_per_batch"]),
                max_padded_tokens=int(config["max_padded_tokens"]),
                amp_enabled=use_amp,
                amp_dtype=amp_dtype,
            )
            if any(
                metric["examples"] != int(config["examples_per_task"])
                for metric in metrics.values()
            ):
                raise ValueError("extrapolation evaluation has the wrong task counts")
            result = {
                **identity,
                "status": "completed",
                "examples": sum(int(metric["examples"]) for metric in metrics.values()),
                "elapsed_seconds": time.monotonic() - started,
                "metrics": metrics,
            }
            _atomic_json(result, output)
            results.append(result)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        manifest = {
            "status": "completed",
            "format_version": "v3-size-extrapolation-evaluation/v1",
            "evaluator_commit": evaluator_commit,
            "data_manifest_sha256": config["data_manifest_sha256"],
            "full_verification": verification,
            "run_count": len(results),
            "task_count": len(tasks),
            "examples_per_task_per_run": config["examples_per_task"],
            "total_model_examples": sum(int(result["examples"]) for result in results),
            "runs": [
                {"run_id": result["run_id"], "result_file": f"per-run/{result['run_id']}.json"}
                for result in results
            ],
        }
        _atomic_json(manifest, output_dir / "manifest.json")
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def _status(task: str, trained_tasks: Sequence[str]) -> str:
    if task in trained_tasks:
        return "seen"
    return "fixed_train_holdout" if task in FIXED_TRAIN_HOLDOUTS else "pool_unseen"


def build_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    per_seed: dict[tuple[str, int, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        per_seed.setdefault(
            (
                str(row["architecture"]),
                int(row["trained_task_count"]),
                str(row["task_status"]),
                int(row["seed"]),
            ),
            [],
        ).append(row)
    seed_macros = []
    for (architecture, task_count, status, seed), group in per_seed.items():
        seed_macros.append(
            {
                "architecture": architecture,
                "trained_task_count": task_count,
                "task_status": status,
                "seed": seed,
                "task_count": len(group),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in (
                        "loss",
                        "token_accuracy",
                        "sequence_accuracy",
                        "in_domain_sequence_accuracy",
                        "sequence_accuracy_delta_out_minus_in",
                    )
                },
            }
        )
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in seed_macros:
        grouped.setdefault(
            (
                str(row["architecture"]),
                int(row["trained_task_count"]),
                str(row["task_status"]),
            ),
            [],
        ).append(row)
    output = []
    for (architecture, task_count, status), group in sorted(grouped.items()):
        if len(group) != 3:
            raise ValueError("extrapolation summary requires three seeds")
        record: dict[str, Any] = {
            "architecture": architecture,
            "trained_task_count": task_count,
            "task_status": status,
            "task_count": int(group[0]["task_count"]),
            "seed_count": 3,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            values = [float(row[metric]) for row in group]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_sample_sd"] = statistics.stdev(values)
        record["in_domain_sequence_accuracy_mean"] = statistics.fmean(
            float(row["in_domain_sequence_accuracy"]) for row in group
        )
        deltas = [float(row["sequence_accuracy_delta_out_minus_in"]) for row in group]
        record["sequence_accuracy_delta_out_minus_in_mean"] = statistics.fmean(deltas)
        record["sequence_accuracy_delta_out_minus_in_sample_sd"] = statistics.stdev(deltas)
        output.append(record)
    return output


def report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, repository = _load_config(config_path)
    output_dir = repository / str(config["output_dir"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("run_count") != 30:
        raise ValueError("size-extrapolation evaluation is incomplete")
    in_domain = repository / str(config["in_domain_evaluation_dir"])
    rows = []
    for entry in manifest["runs"]:
        result = json.loads((output_dir / entry["result_file"]).read_text(encoding="utf-8"))
        baseline = json.loads(
            (in_domain / "per-run" / f"{result['run_id']}.json").read_text(encoding="utf-8")
        )
        for task, metric in result["metrics"].items():
            inside = baseline["metrics"][task]
            row = {
                "run_id": result["run_id"],
                "architecture": result["architecture"],
                "trained_task_count": result["trained_task_count"],
                "seed": result["seed"],
                "task": task,
                "task_status": _status(task, result["trained_tasks"]),
                "examples": metric["examples"],
                "supervised_tokens": metric["tokens"],
                "loss": metric["loss"],
                "token_accuracy": metric["token_accuracy"],
                "sequence_accuracy": metric["sequence_accuracy"],
                "in_domain_loss": inside["loss"],
                "in_domain_token_accuracy": inside["token_accuracy"],
                "in_domain_sequence_accuracy": inside["sequence_accuracy"],
                "loss_delta_out_minus_in": float(metric["loss"]) - float(inside["loss"]),
                "token_accuracy_delta_out_minus_in": float(metric["token_accuracy"]) - float(inside["token_accuracy"]),
                "sequence_accuracy_delta_out_minus_in": float(metric["sequence_accuracy"]) - float(inside["sequence_accuracy"]),
            }
            if not all(math.isfinite(float(row[key])) for key in ("loss", "token_accuracy", "sequence_accuracy")):
                raise ValueError("non-finite size-extrapolation metric")
            rows.append(row)
    if len(rows) != 570:
        raise ValueError("size-extrapolation raw grid is incomplete")
    summary = build_summary(rows)
    result_dir = output_dir.parent
    _atomic_csv(rows, RAW_FIELDS, result_dir / "model_task_results.csv")
    _atomic_csv(summary, SUMMARY_FIELDS, result_dir / "summary.csv")
    lines = [
        "# Permutation-length extrapolation (n=31-40)", "",
        "Thirty completed v3 nested models were evaluated on 1,000 new examples ",
        "per task at lengths 31 through 40. The reduced-word task is excluded ",
        "because some outputs exceed the frozen 1,024-token context. Values below ",
        "are task-macro exact-sequence accuracy, averaged within seed and then ",
        "reported as mean plus/minus sample standard deviation across three seeds.", "",
        "| Architecture | k | Status | n=2-30 | n=31-40 | Change |", "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {'MLP' if row['architecture'] == 'mlp' else 'Transformer'} | {row['trained_task_count']} | {row['task_status']} | "
            f"{100*float(row['in_domain_sequence_accuracy_mean']):.2f}% | "
            f"{100*float(row['sequence_accuracy_mean']):.2f}% +/- {100*float(row['sequence_accuracy_sample_sd']):.2f}% | "
            f"{100*float(row['sequence_accuracy_delta_out_minus_in_mean']):+.2f} pp |"
        )
    lines.extend(["", "This is a post-hoc distribution-shift diagnostic, not a preregistered test. The shift changes both sequence length and the set of atomic number tokens appearing as permutation values.", "", "- [Unaveraged results](model_task_results.csv)", "- [Seed-aggregated summary](summary.csv)", ""])
    _atomic_text("\n".join(lines), result_dir / "README.md")
    report_manifest = {
        "status": "completed",
        "format_version": "v3-size-extrapolation-results/v1",
        "row_count": len(rows),
        "summary_count": len(summary),
        "data_manifest_sha256": config["data_manifest_sha256"],
        "artifacts": {
            name: _sha256(result_dir / name)
            for name in ("model_task_results.csv", "summary.csv", "README.md")
        },
    }
    _atomic_json(report_manifest, result_dir / "manifest.json")
    return report_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (
        evaluate_all(args.config, device_name=args.device)
        if args.command == "evaluate"
        else report(args.config)
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

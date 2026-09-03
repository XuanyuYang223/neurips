"""Validation-only matched-learning-rate sensitivity for Property32 few-shot runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .cka import _atomic_csv, _atomic_json, _atomic_text, _sha256
from .fewshot import _completion_valid, _git_commit, _locked, _run_identity, _train_one
from .property_fewshot import (
    DEFAULT_CONFIG as BASE_CONFIG,
    FORMAT_VERSION as BASE_FORMAT,
    REPLICATE_IDS,
    TASK_COUNTS,
    _base_runs,
    _support_lookup,
    audit_all as audit_primary,
    build_plan,
    load_spec,
    load_support_artifact,
)
from .training import TrainConfig, _default_model_factory


DEFAULT_CONFIG = Path("configs/property32_fewshot_lr_sensitivity.toml")
FORMAT_VERSION = "property32-fewshot-lr-sensitivity/v1"
LOW_LR = 1e-5
HIGH_LR = 3e-4


def _load(config_path: Path = DEFAULT_CONFIG):
    absolute = config_path.resolve()
    repository = absolute.parent.parent
    payload = absolute.read_bytes()
    config = tomllib.loads(payload.decode("utf-8"))
    digest = hashlib.sha256(payload).hexdigest()
    if config.get("protocol_version") != FORMAT_VERSION:
        raise ValueError("unexpected Property32 LR-sensitivity protocol")
    base_path = repository / config["base_config"]
    if _sha256(base_path) != config["base_config_sha256"]:
        raise ValueError("primary Property32 few-shot config changed")
    base = load_spec(base_path)
    if _sha256(base.support_artifact) != config["support_artifact_sha256"]:
        raise ValueError("Property32 support artifact changed")
    expected = {
        "evaluation_split": "validation",
        "shots": 20,
        "max_steps": 200,
        "learning_rates": [LOW_LR, HIGH_LR],
        "new_pretrained_learning_rate": HIGH_LR,
        "new_random_learning_rate": LOW_LR,
        "new_pretrained_runs": 120,
        "new_random_runs": 24,
        "total_new_runs": 144,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Property32 LR-sensitivity design drifted")
    sensitivity = replace(
        base,
        config_path=absolute,
        config_sha256=digest,
        output_dir=(repository / config["output_dir"]).resolve(),
        learning_rate=HIGH_LR,
        random_init_learning_rate=LOW_LR,
    )
    return repository, config, digest, base, sensitivity


def plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    _, _, digest, base, sensitivity = _load(config_path)
    runs = build_plan(base, _base_runs(base, strict=False))
    return {
        "protocol_sha256": digest,
        "run_count": len(runs),
        "pretrained_count": sum(run["initialization"] == "pretrained" for run in runs),
        "random_count": sum(run["initialization"] == "random" for run in runs),
        "completed_count": sum(
            (sensitivity.output_dir / str(run["run_id"]) / "completed.json").is_file()
            for run in runs
        ),
    }


def run_all(config_path: Path = DEFAULT_CONFIG, *, device_name: str | None = None) -> dict[str, Any]:
    repository, _, digest, base, sensitivity = _load(config_path)
    commit = _git_commit(repository)
    artifact = load_support_artifact(base)
    support_sha = _sha256(base.support_artifact)
    support = _support_lookup(artifact)
    runs = build_plan(base, _base_runs(base))
    lock = sensitivity.output_dir / ".controller.lock"
    descriptor = _locked(lock)
    os.close(descriptor)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    markers = []
    try:
        for index, run in enumerate(runs, start=1):
            print(f"LR sensitivity {index}/144: {run['run_id']}", flush=True)
            markers.append(
                _train_one(
                    sensitivity,
                    run,
                    support[(str(run["replicate_id"]), str(run["model_pool"]), str(run["task"]))],
                    support_sha256=support_sha,
                    implementation_commit=commit,
                    device=device,
                    format_version=FORMAT_VERSION,
                )
            )
    finally:
        lock.unlink(missing_ok=True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "completed",
        "protocol_sha256": digest,
        "implementation_commit": commit,
        "run_count": len(markers),
        "validation_model_examples": sum(int(marker["validation"]["examples"]) for marker in markers),
        "runs": [{"run_id": marker["run_id"], "checkpoint_sha256": marker["checkpoint_sha256"]} for marker in markers],
    }
    _atomic_json(manifest, sensitivity.output_dir / "manifest.json")
    return manifest


def audit(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    _, _, digest, base, sensitivity = _load(config_path)
    artifact = load_support_artifact(base)
    support_sha = _sha256(base.support_artifact)
    support = _support_lookup(artifact)
    runs = build_plan(base, _base_runs(base, strict=False))
    passed = []
    incomplete = []
    failed = []
    for run in runs:
        run_dir = sensitivity.output_dir / str(run["run_id"])
        marker_path = run_dir / "completed.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        if not marker_path.is_file() or not checkpoint_path.is_file():
            incomplete.append(str(run["run_id"]))
            continue
        issues = []
        try:
            marker = json.loads(marker_path.read_text())
            records = support[(str(run["replicate_id"]), str(run["model_pool"]), str(run["task"]))]
            identity = _run_identity(
                sensitivity,
                run,
                support_sha256=support_sha,
                support_ids=[int(record["id"]) for record in records],
                implementation_commit=str(marker.get("implementation_commit")),
                format_version=FORMAT_VERSION,
            )
            if not _completion_valid(marker_path, identity, checkpoint_path):
                issues.append("completion identity or checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                issues.append("checkpoint identity mismatch")
            model_config = TrainConfig.from_value(checkpoint["base_train_config"])
            model = _default_model_factory(model_config)
            model.load_state_dict(checkpoint["model"], strict=True)
            if not all(torch.isfinite(tensor).all().item() for tensor in checkpoint["model"].values()):
                issues.append("non-finite model tensor")
            validation = checkpoint.get("validation", {})
            if validation != marker.get("validation") or validation.get("examples") != 5_000:
                issues.append("validation identity or count mismatch")
            expected_lr = (HIGH_LR if run["initialization"] == "pretrained" else LOW_LR) * sensitivity.min_learning_rate_ratio
            if not math.isclose(float(checkpoint.get("final_learning_rate", -1)), expected_lr, rel_tol=1e-9):
                issues.append("final learning rate mismatch")
            del model, checkpoint
        except Exception as error:
            issues.append(f"{type(error).__name__}: {error}")
        if issues:
            failed.append({"run_id": run["run_id"], "issues": issues})
        else:
            passed.append(str(run["run_id"]))
    ok = len(passed) == 144 and not failed
    return {
        "format_version": FORMAT_VERSION,
        "status": "passed" if ok else "failed" if failed else "incomplete",
        "ok": ok,
        "protocol_sha256": digest,
        "passed_count": len(passed),
        "incomplete_count": len(incomplete),
        "failed_count": len(failed),
        "incomplete": incomplete,
        "failed": failed,
    }


def _rows_for_spec(spec, source: str, runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        marker = json.loads((spec.output_dir / str(run["run_id"]) / "completed.json").read_text())
        metric = marker["validation"]
        lr = spec.learning_rate if run["initialization"] == "pretrained" else spec.random_init_learning_rate
        rows.append(
            {
                "source": source,
                "initialization": run["initialization"],
                "learning_rate": lr,
                "replicate_id": run["replicate_id"],
                "model_pool": run["model_pool"],
                "base_trained_task_count": run["base_trained_task_count"],
                "seed": run["seed"],
                "task": run["task"],
                "target_family": run["target_family"],
                "loss": metric["loss"],
                "token_accuracy": metric["token_accuracy"],
                "sequence_accuracy": metric["sequence_accuracy"],
                "examples": metric["examples"],
                "checkpoint_sha256": marker["checkpoint_sha256"],
            }
        )
    return rows


def export_results(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    repository, config, digest, base, sensitivity = _load(config_path)
    if not audit_primary(base.config_path)["ok"] or not audit(config_path)["ok"]:
        raise ValueError("both primary and sensitivity checkpoints must pass audit")
    runs = build_plan(base, _base_runs(base, strict=False))
    rows = _rows_for_spec(base, "primary", runs) + _rows_for_spec(sensitivity, "sensitivity", runs)
    lookup = {
        (row["initialization"], float(row["learning_rate"]), row["replicate_id"], row["model_pool"], int(row["base_trained_task_count"]), row["task"]): row
        for row in rows
    }
    summaries = []
    for learning_rate in (LOW_LR, HIGH_LR):
        for task_count in TASK_COUNTS:
            replicate_values = []
            for replicate_id in REPLICATE_IDS:
                warm = []
                random = []
                for pool in ("a", "b"):
                    targets = base.target_sets[(replicate_id, pool)]
                    warm.extend(lookup[("pretrained", learning_rate, replicate_id, pool, task_count, task)] for task in targets)
                    random.extend(lookup[("random", learning_rate, replicate_id, pool, 0, task)] for task in targets)
                replicate_values.append(
                    {
                        "warm": statistics.fmean(float(row["sequence_accuracy"]) for row in warm),
                        "random": statistics.fmean(float(row["sequence_accuracy"]) for row in random),
                    }
                )
            warm_values = [row["warm"] for row in replicate_values]
            random_values = [row["random"] for row in replicate_values]
            contrast = [warm - random for warm, random in zip(warm_values, random_values, strict=True)]
            summaries.append(
                {
                    "learning_rate": learning_rate,
                    "base_trained_task_count": task_count,
                    "replicate_count": 3,
                    "pretrained_sequence_accuracy_mean": statistics.fmean(warm_values),
                    "pretrained_sequence_accuracy_sample_sd": statistics.stdev(warm_values),
                    "random_sequence_accuracy_mean": statistics.fmean(random_values),
                    "random_sequence_accuracy_sample_sd": statistics.stdev(random_values),
                    "pretrained_minus_random_mean": statistics.fmean(contrast),
                    "pretrained_minus_random_sample_sd": statistics.stdev(contrast),
                }
            )
    output = repository / config["results_dir"]
    _atomic_csv(rows, tuple(rows[0]), output / "validation_model_results.csv")
    _atomic_csv(summaries, tuple(summaries[0]), output / "matched_lr_summary.csv")
    lines = [
        "# Property32 twenty-shot matched-learning-rate sensitivity",
        "",
        "This post-hoc analysis uses validation only. Values are exact sequence accuracy mean +/- sample SD over three joint task-split/model-seed replicates.",
        "",
        "| Learning rate | k | Pretrained | Random init | Pretrained minus random |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['learning_rate']:.0e} | {row['base_trained_task_count']} | "
            f"{100 * row['pretrained_sequence_accuracy_mean']:.2f}% +/- {100 * row['pretrained_sequence_accuracy_sample_sd']:.2f}% | "
            f"{100 * row['random_sequence_accuracy_mean']:.2f}% +/- {100 * row['random_sequence_accuracy_sample_sd']:.2f}% | "
            f"{100 * row['pretrained_minus_random_mean']:+.2f} +/- {100 * row['pretrained_minus_random_sample_sd']:.2f} pp |"
        )
    lines.extend(["", "The learning-rate sweep was specified after the primary test result and is not a new confirmatory test-set result.", ""])
    _atomic_text("\n".join(lines), output / "README.md")
    manifest = {
        "format_version": "property32-fewshot-lr-sensitivity-results/v1",
        "status": "completed",
        "protocol_sha256": digest,
        "row_count": len(rows),
        "summary_row_count": len(summaries),
        "artifacts": {name: _sha256(output / name) for name in ("validation_model_results.csv", "matched_lr_summary.csv", "README.md")},
    }
    _atomic_json(manifest, output / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "audit", "results"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "plan":
        value = plan(args.config)
    elif args.action == "run":
        value = run_all(args.config, device_name=args.device)
    elif args.action == "audit":
        value = audit(args.config)
    else:
        value = export_results(args.config)
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0 if value.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

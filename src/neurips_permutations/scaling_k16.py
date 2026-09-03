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
from xml.sax.saxutils import escape

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


def _factorial_effect_rows(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    """Return seed-paired data, depth, and interaction contrasts."""

    lookup = {
        (str(row["condition"]), str(row["architecture"]), int(row["seed"])): row
        for row in rows
    }
    contrasts = {
        "data_effect_at_1x_model": ("data10x_model1x", "baseline"),
        "model_effect_at_1x_data": ("data1x_model2x", "baseline"),
        "data_effect_at_2x_model": ("data10x_model2x", "data1x_model2x"),
        "model_effect_at_10x_data": ("data10x_model2x", "data10x_model1x"),
    }
    effects: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for metric in metrics:
            for name, (left, right) in contrasts.items():
                values = [
                    float(lookup[left, architecture, seed][metric])
                    - float(lookup[right, architecture, seed][metric])
                    for seed in MODEL_SEEDS
                ]
                mean, sd = _mean_sd(values)
                effects.append(
                    {
                        "architecture": architecture,
                        "metric": metric,
                        "contrast": name,
                        "mean": mean,
                        "sample_sd": sd,
                    }
                )
            interaction = [
                float(lookup["data10x_model2x", architecture, seed][metric])
                - float(lookup["data10x_model1x", architecture, seed][metric])
                - float(lookup["data1x_model2x", architecture, seed][metric])
                + float(lookup["baseline", architecture, seed][metric])
                for seed in MODEL_SEEDS
            ]
            mean, sd = _mean_sd(interaction)
            effects.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "contrast": "data_by_model_interaction",
                    "mean": mean,
                    "sample_sd": sd,
                }
            )
    return effects


def _scaling_figure_svg(summary: Sequence[Mapping[str, Any]]) -> str:
    """Render a dependency-free grouped bar chart with sample-SD error bars."""

    lookup = {
        (str(row["condition"]), str(row["architecture"])): row for row in summary
    }
    if set(lookup) != {
        (condition, architecture)
        for condition in CONDITION_ORDER
        for architecture in ARCHITECTURES
    }:
        raise ValueError("scaling figure requires the complete 4 x 2 grid")
    values = [
        float(row["structured_holdout_sequence_accuracy_mean"])
        + float(row["structured_holdout_sequence_accuracy_sample_sd"])
        for row in summary
    ]
    y_max = max(0.01, math.ceil(max(values) * 20.0) / 20.0)
    width, height = 960, 560
    left, right, top, bottom = 88, 32, 52, 128
    plot_width = width - left - right
    plot_height = height - top - bottom

    def y(value: float) -> float:
        return top + plot_height * (1.0 - max(0.0, min(value, y_max)) / y_max)

    colors = {"transformer": "#3b82f6", "mlp": "#f59e0b"}
    labels = {
        "baseline": ("1x data", "1x depth"),
        "data10x_model1x": ("10x data", "1x depth"),
        "data1x_model2x": ("1x data", "2x depth"),
        "data10x_model2x": ("10x data", "2x depth"),
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1}.error{stroke:#111827;stroke-width:2}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="20" font-weight="700">k=16 structured-holdout exact accuracy</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        position = y(value)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{position:.2f}" x2="{width-right}" y2="{position:.2f}"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{position+5:.2f}" text-anchor="end" font-size="13">{100*value:.1f}%</text>'
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}"/>',
        ]
    )
    group_width = plot_width / len(CONDITION_ORDER)
    bar_width = min(62.0, group_width * 0.28)
    for condition_index, condition in enumerate(CONDITION_ORDER):
        center = left + group_width * (condition_index + 0.5)
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            row = lookup[condition, architecture]
            mean = float(row["structured_holdout_sequence_accuracy_mean"])
            sd = float(row["structured_holdout_sequence_accuracy_sample_sd"])
            bar_center = center + (-0.58 if architecture_index == 0 else 0.58) * bar_width
            bar_top = y(mean)
            bar_height = top + plot_height - bar_top
            lines.append(
                f'<rect x="{bar_center-bar_width/2:.2f}" y="{bar_top:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{colors[architecture]}"/>'
            )
            error_top, error_bottom = y(mean + sd), y(max(0.0, mean - sd))
            lines.extend(
                [
                    f'<line class="error" x1="{bar_center:.2f}" y1="{error_top:.2f}" x2="{bar_center:.2f}" y2="{error_bottom:.2f}"/>',
                    f'<line class="error" x1="{bar_center-7:.2f}" y1="{error_top:.2f}" x2="{bar_center+7:.2f}" y2="{error_top:.2f}"/>',
                    f'<line class="error" x1="{bar_center-7:.2f}" y1="{error_bottom:.2f}" x2="{bar_center+7:.2f}" y2="{error_bottom:.2f}"/>',
                ]
            )
        first, second = labels[condition]
        lines.extend(
            [
                f'<text x="{center:.2f}" y="{top+plot_height+30}" text-anchor="middle" font-size="14">{escape(first)}</text>',
                f'<text x="{center:.2f}" y="{top+plot_height+50}" text-anchor="middle" font-size="14">{escape(second)}</text>',
            ]
        )
    legend_y = height - 28
    for index, architecture in enumerate(ARCHITECTURES):
        x = width / 2 - 145 + index * 190
        lines.append(
            f'<rect x="{x:.2f}" y="{legend_y-13}" width="18" height="18" fill="{colors[architecture]}"/>'
        )
        lines.append(
            f'<text x="{x+27:.2f}" y="{legend_y+1}" font-size="14">{"Transformer" if architecture == "transformer" else "MLP"}</text>'
        )
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def _paper_section(
    summary: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]]
) -> str:
    exact_effects = [
        row
        for row in effects
        if row["metric"] == "structured_holdout_sequence_accuracy"
    ]
    effect_lookup = {
        (str(row["architecture"]), str(row["contrast"])): row
        for row in exact_effects
    }
    lines = [
        "# k=16 Scaling Study: Paper-Ready Section",
        "",
        "## Methods",
        "",
        "We tested whether weak zero-shot execution of structured held-out permutation operations was limited by training exposure or model depth. At the fully populated `k=16` endpoint, we crossed 1x versus 10x training exposure with 1x versus 2x depth. The 1x condition used 1.28 million examples, four Transformer layers or one causal-MLP block, and 20,000 optimizer updates. The 10x exposure condition used 12.8 million examples and 200,000 updates; the 2x-depth condition used eight Transformer layers or two causal-MLP blocks. All other optimizer, tokenizer, task-mixture, and sequence-length settings were fixed.",
        "",
        "Each factorial cell contains seeds 17, 42, and 314159 for both architectures. The primary outcome is exact complete-answer accuracy, macro-averaged within each model over three structured tasks held out from training: reduced-word translation, composition, and Lehmer-code translation. Parity is reported separately as a short Boolean diagnostic. Every final model was evaluated on the same frozen v3 test split with 5,000 examples per task. We first formed a three-task macro within each seed and then report the mean and sample standard deviation over the three paired seeds.",
        "",
        "## Results",
        "",
        "| Architecture | Data | Depth | Structured loss | Token accuracy | Exact accuracy | Parity exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {'Transformer' if row['architecture'] == 'transformer' else 'MLP'} | {row['data_multiplier']}x | {row['model_multiplier']}x | "
            f"{row['structured_holdout_loss_mean']:.4f} +/- {row['structured_holdout_loss_sample_sd']:.4f} | "
            f"{100*row['structured_holdout_token_accuracy_mean']:.3f}% +/- {100*row['structured_holdout_token_accuracy_sample_sd']:.3f}% | "
            f"{100*row['structured_holdout_sequence_accuracy_mean']:.3f}% +/- {100*row['structured_holdout_sequence_accuracy_sample_sd']:.3f}% | "
            f"{100*row['parity_sequence_accuracy_mean']:.3f}% +/- {100*row['parity_sequence_accuracy_sample_sd']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "Seed-paired exact-accuracy effects are:",
            "",
            "| Architecture | 10x data at 1x depth | 2x depth at 1x data | 10x data at 2x depth | 2x depth at 10x data | Interaction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    names = (
        "data_effect_at_1x_model",
        "model_effect_at_1x_data",
        "data_effect_at_2x_model",
        "model_effect_at_10x_data",
        "data_by_model_interaction",
    )
    for architecture in ARCHITECTURES:
        rendered = []
        for name in names:
            row = effect_lookup[architecture, name]
            rendered.append(
                f"{100*float(row['mean']):+.3f} +/- {100*float(row['sample_sd']):.3f} pp"
            )
        label = "Transformer" if architecture == "transformer" else "MLP"
        lines.append(f"| {label} | " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "These contrasts are descriptive estimates from three paired seeds. Positive accuracy contrasts indicate improvement; negative loss contrasts indicate improvement. The complete loss and token-accuracy contrasts are retained in `factorial_effects.csv`.",
            "",
            "## Limitations",
            "",
            "1. The experiment has only three seeds per cell, so the sample standard deviations quantify observed seed variability but do not support precise population-level inference.",
            "2. The study fixes `k=16`; it tests one difficult endpoint rather than a scaling law across task counts.",
            "3. The 10x-data intervention also uses 10x more optimizer updates and approximately 10x more training examples. It is therefore an exposure-and-compute intervention, not a pure corpus-size intervention at matched compute.",
            "4. The 2x-model intervention doubles depth, not every architectural dimension; its parameter multiplier is architecture dependent.",
            "5. The primary macro contains three structured outputs with different lengths and difficulties. Parity is excluded from that macro because a Boolean answer is not directly comparable.",
            "6. The experiment measures exact behavioral transfer to task tokens held out from gradient updates. It does not identify whether failure comes from ungrounded task-token semantics, missing algorithms, or decoding errors.",
            "7. Teacher-forced token accuracy and loss can improve through formatting or local-token prediction even when complete-answer exact accuracy remains low.",
            "",
        ]
    )
    return "\n".join(lines)


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
    effect_metrics = (
        "structured_holdout_loss",
        "structured_holdout_token_accuracy",
        "structured_holdout_sequence_accuracy",
    )
    effects = _factorial_effect_rows(rows, effect_metrics)
    output = repository / config["results_dir"]
    raw_fields = tuple(rows[0])
    summary_fields = tuple(summary[0])
    effect_fields = ("architecture", "metric", "contrast", "mean", "sample_sd")
    _atomic_csv(rows, raw_fields, output / "model_results.csv")
    _atomic_csv(summary, summary_fields, output / "summary.csv")
    _atomic_csv(effects, effect_fields, output / "factorial_effects.csv")
    _atomic_text(_scaling_figure_svg(summary), output / "structured_exact_accuracy.svg")
    _atomic_text(_paper_section(summary, effects), output / "PAPER_SECTION.md")
    lines = [
        "# k=16 data and model scaling results",
        "",
        "The primary metric is exact sequence accuracy macro-averaged over `to_reduced_word`, `compose`, and `to_lehmer`. Values are mean +/- sample SD over three paired model seeds.",
        "",
        "| Architecture | Condition | Structured loss | Structured token | Structured exact | Parity exact |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['architecture']} | `{row['condition']}` | "
            f"{row['structured_holdout_loss_mean']:.4f} +/- {row['structured_holdout_loss_sample_sd']:.4f} | "
            f"{100 * row['structured_holdout_token_accuracy_mean']:.3f}% +/- {100 * row['structured_holdout_token_accuracy_sample_sd']:.3f}% | "
            f"{100 * row['structured_holdout_sequence_accuracy_mean']:.3f}% +/- {100 * row['structured_holdout_sequence_accuracy_sample_sd']:.3f}% | "
            f"{100 * row['parity_sequence_accuracy_mean']:.3f}% +/- {100 * row['parity_sequence_accuracy_sample_sd']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "![Structured holdout scaling](structured_exact_accuracy.svg)",
            "",
            "The table is descriptive with only three seeds. See `factorial_effects.csv` for seed-paired changes in structured loss, teacher-forced token accuracy, and exact accuracy. For loss, a negative contrast is an improvement; for accuracy, a positive contrast is an improvement. See `model_results.csv` for every endpoint and `PAPER_SECTION.md` for paper-ready Methods, Results, and Limitations.",
            "",
        ]
    )
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
            for name in (
                "model_results.csv",
                "summary.csv",
                "factorial_effects.csv",
                "structured_exact_accuracy.svg",
                "PAPER_SECTION.md",
                "README.md",
            )
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

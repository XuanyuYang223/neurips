"""Aggregate the paired 5/20/100-shot permutation adaptation curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .cka import _atomic_json, _atomic_text
from .fewshot import DEFAULT_CONFIG, _sha256, load_spec, load_support_artifact
from .fewshot_results import (
    GAIN_FIELDS,
    RAW_FIELDS,
    SUMMARY_FIELDS,
    TASK_SUMMARY_FIELDS,
    _write_csv,
    build_gains,
    build_summary,
    build_task_summary,
    load_rows,
)


DEFAULT_CONFIGS = {
    5: Path("configs/henry_permutation_fewshot_shot05.toml"),
    20: DEFAULT_CONFIG,
    100: Path("configs/henry_permutation_fewshot_shot100.toml"),
}
DEFAULT_OUTPUT_DIR = Path("results/v3/fewshot/shot-curve")

CURVE_SUMMARY_FIELDS = ("shots", *SUMMARY_FIELDS)
CURVE_TASK_SUMMARY_FIELDS = ("shots", *TASK_SUMMARY_FIELDS)
CURVE_GAIN_FIELDS = ("shots", *GAIN_FIELDS)
DELTA_FIELDS = (
    "initialization",
    "architecture",
    "base_trained_task_count",
    "seed_count",
    "task_count",
    "loss_100_minus_5_mean",
    "loss_100_minus_5_sample_sd",
    "token_accuracy_100_minus_5_mean",
    "token_accuracy_100_minus_5_sample_sd",
    "sequence_accuracy_100_minus_5_mean",
    "sequence_accuracy_100_minus_5_sample_sd",
)


def _validate_nested_supports(configs: Mapping[int, Path]) -> dict[int, str]:
    if set(configs) != {5, 20, 100}:
        raise ValueError("shot curve requires exactly the 5/20/100 endpoints")
    artifacts: dict[int, dict[str, Any]] = {}
    hashes: dict[int, str] = {}
    for shots, config_path in configs.items():
        spec = load_spec(config_path)
        if spec.shots != shots:
            raise ValueError("shot count does not match its curve endpoint")
        artifacts[shots] = load_support_artifact(spec)
        hashes[shots] = _sha256(spec.support_artifact)
    by_shots = {
        shots: {str(item["key"]): list(item["record_ids"]) for item in artifact["sets"]}
        for shots, artifact in artifacts.items()
    }
    if not all(set(by_shots[shots]) == set(by_shots[20]) for shots in (5, 100)):
        raise ValueError("shot-curve support cells differ")
    for key, anchor_ids in by_shots[20].items():
        if by_shots[5][key] != anchor_ids[:5]:
            raise ValueError(f"5-shot cell is not nested in 20-shot cell: {key}")
        if by_shots[100][key][:20] != anchor_ids:
            raise ValueError(f"20-shot cell is not nested in 100-shot cell: {key}")
    return hashes


def build_endpoint_delta(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate matched 100-shot minus 5-shot changes across three seeds."""

    endpoints: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if int(row["shots"]) not in {5, 100}:
            continue
        key = (
            int(row["shots"]),
            str(row["initialization"]),
            str(row["architecture"]),
            int(row["base_trained_task_count"]),
            int(row["seed"]),
            str(row["task"]),
        )
        if key in endpoints:
            raise ValueError("duplicate shot-curve endpoint row")
        endpoints[key] = row
    seed_values: dict[tuple[str, str, int, int], dict[str, float]] = {}
    conditions = {
        key[1:5] for key in endpoints if key[0] == 5
    }
    for initialization, architecture, task_count, seed in conditions:
        selected = []
        for task in sorted(
            key[5]
            for key in endpoints
            if key[0] == 5 and key[1:5] == (
                initialization,
                architecture,
                task_count,
                seed,
            )
        ):
            left = endpoints[(5, initialization, architecture, task_count, seed, task)]
            right = endpoints.get(
                (100, initialization, architecture, task_count, seed, task)
            )
            if right is None:
                raise ValueError("100-shot endpoint lacks a paired 5-shot row")
            selected.append((left, right))
        if len(selected) != 4:
            raise ValueError("each endpoint delta requires four holdout tasks")
        seed_values[(initialization, architecture, task_count, seed)] = {
            metric: statistics.fmean(
                float(right[metric]) - float(left[metric])
                for left, right in selected
            )
            for metric in ("loss", "token_accuracy", "sequence_accuracy")
        }
    grouped: dict[tuple[str, str, int], list[Mapping[str, float]]] = {}
    for (initialization, architecture, task_count, _), values in seed_values.items():
        grouped.setdefault((initialization, architecture, task_count), []).append(values)
    output: list[dict[str, Any]] = []
    for (initialization, architecture, task_count), values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError("endpoint delta requires three paired seeds")
        row: dict[str, Any] = {
            "initialization": initialization,
            "architecture": architecture,
            "base_trained_task_count": task_count,
            "seed_count": 3,
            "task_count": 4,
        }
        for metric in ("loss", "token_accuracy", "sequence_accuracy"):
            observed = [float(value[metric]) for value in values]
            row[f"{metric}_100_minus_5_mean"] = statistics.fmean(observed)
            row[f"{metric}_100_minus_5_sample_sd"] = statistics.stdev(observed)
        output.append(row)
    return output


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_readme(
    summary: Sequence[Mapping[str, Any]],
    structured: Sequence[Mapping[str, Any]],
    deltas: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Paired 5/20/100-shot adaptation curve",
        "",
        "This post-hoc extension varies only the number of support examples. The ",
        "support sets are strictly nested (`5` is a subset of `20`, which is a ",
        "subset of `100`), and every endpoint uses the same 200 update steps, ",
        "learning rates, four v3 holdout tasks, architectures, base models, and ",
        "three model seeds. Each reported value is a task macro within seed, then ",
        "mean plus/minus sample standard deviation across seeds.",
        "",
        "## Exact-sequence accuracy across all four holdouts",
        "",
        "| Architecture | Base k | 5-shot | 20-shot | 100-shot |",
        "|---|---:|---:|---:|---:|",
    ]
    pretrained = {
        (str(row["architecture"]), int(row["base_trained_task_count"]), int(row["shots"])): row
        for row in summary
        if row["initialization"] == "pretrained"
    }
    for architecture in ("transformer", "mlp"):
        for task_count in (1, 2, 4, 8, 16):
            cells = []
            for shots in (5, 20, 100):
                row = pretrained[(architecture, task_count, shots)]
                cells.append(
                    f"{_pct(float(row['sequence_accuracy_mean']))} +/- "
                    f"{_pct(float(row['sequence_accuracy_sample_sd']))}"
                )
            lines.append(
                f"| {architecture.title()} | {task_count} | "
                + " | ".join(cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Structured-output exact accuracy",
            "",
            "This table excludes the scalar parity task and averages reduced word, ",
            "composition, and Lehmer translation.",
            "",
            "| Architecture | Base k | 5-shot | 20-shot | 100-shot |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    structured_rows = {
        (str(row["architecture"]), int(row["base_trained_task_count"]), int(row["shots"])): row
        for row in structured
        if row["initialization"] == "pretrained"
    }
    for architecture in ("transformer", "mlp"):
        for task_count in (1, 2, 4, 8, 16):
            values = [
                _pct(float(structured_rows[(architecture, task_count, shots)]["sequence_accuracy_mean"]))
                for shots in (5, 20, 100)
            ]
            lines.append(
                f"| {architecture.title()} | {task_count} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary question is whether increasing support from 5 to 100 ",
            "examples improves complete-answer accuracy more for pretrained models ",
            "than for matched randomly initialized controls. A larger token accuracy ",
            "alone is insufficient because delimiters and copied tokens can dominate ",
            "that metric. The 20-shot test result was inspected before this extension, ",
            "so the full curve is a fixed post-hoc robustness analysis rather than a ",
            "new untouched confirmatory test.",
            "",
            "Observed support-size result: increasing support from 5 to 100 examples did not produce a consistent pretrained-model improvement. The largest Transformer all-task exact gain was about 2.55 percentage points at base k=4; the other k values changed much less or decreased. Exact accuracy on the three structured outputs remained essentially zero. Matched random-initialization controls also improved with more unique examples, so the curve does not provide strong evidence that pretraining created a robust few-shot operation learner.",
            "",
            "All endpoints use 800 training presentations (200 steps times batch size ",
            "4). Consequently, each support example is reused about 160, 40, or 8 ",
            "times at 5, 20, or 100 shots. This controls update compute but intentionally ",
            "does not control repetitions per unique example.",
            "",
            "## Artifacts",
            "",
            "- [All unaveraged model-task endpoints](model_task_results.csv)",
            "- [All-task endpoint summary](summary.csv)",
            "- [Structured-output summary](structured_summary.csv)",
            "- [Per-task summary](task_summary.csv)",
            "- [Paired adaptation gains](adaptation_gains.csv)",
            "- [Matched 100-shot minus 5-shot changes](endpoint_delta.csv)",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def run_curve(
    configs: Mapping[int, Path] = DEFAULT_CONFIGS,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    zero_shot_evaluation_dir: Path = Path("results/v3/evaluation"),
) -> dict[str, Any]:
    support_hashes = _validate_nested_supports(configs)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    structured: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    gains: list[dict[str, Any]] = []
    config_hashes: dict[int, str] = {}
    for shots in (5, 20, 100):
        config_path = configs[shots]
        spec = load_spec(config_path)
        config_hashes[shots] = spec.config_sha256
        rows = load_rows(
            config_path, zero_shot_evaluation_dir=zero_shot_evaluation_dir
        )
        if any(int(row["shots"]) != shots for row in rows):
            raise ValueError("result row has the wrong shot endpoint")
        all_rows.extend(rows)
        for destination, records in (
            (summaries, build_summary(rows)),
            (
                structured,
                build_summary(
                    rows, tasks=("to_reduced_word", "compose", "to_lehmer")
                ),
            ),
            (task_summaries, build_task_summary(rows)),
            (gains, build_gains(rows)),
        ):
            destination.extend({"shots": shots, **record} for record in records)
    if len(all_rows) != 432:
        raise ValueError("shot curve must contain 432 unaveraged endpoints")
    deltas = build_endpoint_delta(all_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_task_results.csv": (all_rows, RAW_FIELDS),
        "summary.csv": (summaries, CURVE_SUMMARY_FIELDS),
        "structured_summary.csv": (structured, CURVE_SUMMARY_FIELDS),
        "task_summary.csv": (task_summaries, CURVE_TASK_SUMMARY_FIELDS),
        "adaptation_gains.csv": (gains, CURVE_GAIN_FIELDS),
        "endpoint_delta.csv": (deltas, DELTA_FIELDS),
    }
    for name, (records, fields) in outputs.items():
        _write_csv(output_dir / name, records, fields)
    readme = _render_readme(summaries, structured, deltas)
    _atomic_text(readme, output_dir / "README.md")
    manifest = {
        "status": "completed",
        "protocol_version": "henry-permutation-fewshot-shot-curve-results/v1",
        "shots": [5, 20, 100],
        "run_count": 432,
        "config_sha256": {str(key): value for key, value in config_hashes.items()},
        "support_artifact_sha256": {
            str(key): value for key, value in support_hashes.items()
        },
        "test_manifest_sha256": load_spec(configs[20]).test_manifest_sha256,
        "artifacts": {
            name: _sha256(output_dir / name) for name in (*outputs, "README.md")
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-5", type=Path, default=DEFAULT_CONFIGS[5])
    parser.add_argument("--config-20", type=Path, default=DEFAULT_CONFIGS[20])
    parser.add_argument("--config-100", type=Path, default=DEFAULT_CONFIGS[100])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--zero-shot-evaluation-dir",
        type=Path,
        default=Path("results/v3/evaluation"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_curve(
        {5: args.config_5, 20: args.config_20, 100: args.config_100},
        output_dir=args.output_dir,
        zero_shot_evaluation_dir=args.zero_shot_evaluation_dir,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

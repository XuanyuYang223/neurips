"""Export and summarize the frozen Property32 twenty-shot experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .cka import _atomic_csv, _atomic_json, _atomic_text, _git_commit, _sha256
from .property_fewshot import (
    DEFAULT_CONFIG,
    REPLICATE_IDS,
    TASK_COUNTS,
    TEST_FORMAT_VERSION,
    _base_runs,
    audit_all,
    build_plan,
    load_spec,
)


DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/fewshot")
METRICS = ("loss", "token_accuracy", "sequence_accuracy")
CONTRASTS = (
    "loss_improvement_from_zero_shot",
    "token_accuracy_improvement_from_zero_shot",
    "sequence_accuracy_improvement_from_zero_shot",
    "loss_improvement_over_random",
    "token_accuracy_improvement_over_random",
    "sequence_accuracy_improvement_over_random",
)
RAW_FIELDS = (
    "run_id",
    "initialization",
    "replicate_id",
    "model_pool",
    "base_run_id",
    "base_trained_task_count",
    "seed",
    "task",
    "target_family",
    "shots",
    "evaluation_split",
    "examples",
    "supervised_tokens",
    *METRICS,
    "zero_shot_loss",
    "zero_shot_token_accuracy",
    "zero_shot_sequence_accuracy",
    *CONTRASTS,
    "checkpoint_sha256",
    "fewshot_config_sha256",
    "support_artifact_sha256",
    "test_manifest_sha256",
    "test_source_shard_sha256",
    "evaluator_commit",
)
REPLICATE_FIELDS = (
    "replicate_id",
    "model_seed",
    "base_trained_task_count",
    "pool_direction_count",
    "target_count",
    *METRICS,
    *CONTRASTS,
)
SUMMARY_FIELDS = (
    "base_trained_task_count",
    "replicate_count",
    *tuple(
        name
        for metric in (*METRICS, *CONTRASTS)
        for name in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)
FAMILY_FIELDS = (
    "base_trained_task_count",
    "target_family",
    "replicate_count",
    *tuple(
        name
        for metric in (*METRICS, *CONTRASTS)
        for name in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)
RANDOM_FIELDS = (
    "replicate_count",
    *tuple(
        name
        for metric in METRICS
        for name in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def load_rows(config_path: Path = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    spec = load_spec(config_path)
    audit = audit_all(config_path)
    if not audit["ok"]:
        raise ValueError("Property32 few-shot checkpoints must pass strict audit")
    manifest_path = spec.evaluation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format_version": TEST_FORMAT_VERSION,
        "status": "completed",
        "fewshot_config_sha256": spec.config_sha256,
        "support_artifact_sha256": _sha256(spec.support_artifact),
        "test_manifest_sha256": spec.test_manifest_sha256,
        "test_source_shard_index": spec.test_source_shard_index,
        "test_source_shard_sha256": spec.test_source_shard_sha256,
        "zero_shot_model_count": 30,
        "adapted_run_count": 144,
        "examples_per_task": 2_500,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Property32 few-shot evaluation manifest identity differs")
    base_runs = _base_runs(spec, strict=False)
    plan = build_plan(spec, base_runs)
    zero: dict[str, Mapping[str, Any]] = {}
    for name in manifest["zero_shot_results"]:
        value = json.loads((spec.evaluation_dir / name).read_text(encoding="utf-8"))
        zero[str(value["run_id"])] = value["metrics"]
    if len(zero) != 30:
        raise ValueError("Property32 zero-shot result grid is incomplete")
    rows: list[dict[str, Any]] = []
    by_run = {str(run["run_id"]): run for run in plan}
    for name in manifest["adapted_results"]:
        value = json.loads((spec.evaluation_dir / name).read_text(encoding="utf-8"))
        run = by_run[str(value["run_id"])]
        metric = value["metrics"][str(run["task"])]
        row: dict[str, Any] = {
            "run_id": run["run_id"],
            "initialization": run["initialization"],
            "replicate_id": run["replicate_id"],
            "model_pool": run["model_pool"],
            "base_run_id": run["base_run_id"],
            "base_trained_task_count": run["base_trained_task_count"],
            "seed": run["seed"],
            "task": run["task"],
            "target_family": run["target_family"],
            "shots": spec.shots,
            "evaluation_split": "property_source_shard_199",
            "examples": metric["examples"],
            "supervised_tokens": metric["tokens"],
            "loss": metric["loss"],
            "token_accuracy": metric["token_accuracy"],
            "sequence_accuracy": metric["sequence_accuracy"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "fewshot_config_sha256": value["fewshot_config_sha256"],
            "support_artifact_sha256": manifest["support_artifact_sha256"],
            "test_manifest_sha256": value["test_manifest_sha256"],
            "test_source_shard_sha256": value["test_source_shard_sha256"],
            "evaluator_commit": value["evaluator_commit"],
        }
        if run["initialization"] == "pretrained":
            baseline = zero[str(run["base_run_id"])][str(run["task"])]
            row.update(
                {
                    "zero_shot_loss": baseline["loss"],
                    "zero_shot_token_accuracy": baseline["token_accuracy"],
                    "zero_shot_sequence_accuracy": baseline["sequence_accuracy"],
                    "loss_improvement_from_zero_shot": float(baseline["loss"]) - float(metric["loss"]),
                    "token_accuracy_improvement_from_zero_shot": float(metric["token_accuracy"]) - float(baseline["token_accuracy"]),
                    "sequence_accuracy_improvement_from_zero_shot": float(metric["sequence_accuracy"]) - float(baseline["sequence_accuracy"]),
                }
            )
        else:
            row.update(
                {
                    "zero_shot_loss": "",
                    "zero_shot_token_accuracy": "",
                    "zero_shot_sequence_accuracy": "",
                    "loss_improvement_from_zero_shot": "",
                    "token_accuracy_improvement_from_zero_shot": "",
                    "sequence_accuracy_improvement_from_zero_shot": "",
                }
            )
        rows.append(row)
    if len(rows) != 144 or len({row["run_id"] for row in rows}) != 144:
        raise ValueError("Property32 few-shot result rows are incomplete")
    random = {
        (row["replicate_id"], row["model_pool"], row["seed"], row["task"]): row
        for row in rows
        if row["initialization"] == "random"
    }
    if len(random) != 24:
        raise ValueError("Property32 random-control grid is incomplete")
    for row in rows:
        if row["initialization"] == "pretrained":
            control = random[(row["replicate_id"], row["model_pool"], row["seed"], row["task"])]
            row.update(
                {
                    "loss_improvement_over_random": float(control["loss"]) - float(row["loss"]),
                    "token_accuracy_improvement_over_random": float(row["token_accuracy"]) - float(control["token_accuracy"]),
                    "sequence_accuracy_improvement_over_random": float(row["sequence_accuracy"]) - float(control["sequence_accuracy"]),
                }
            )
        else:
            row.update(
                {
                    "loss_improvement_over_random": "",
                    "token_accuracy_improvement_over_random": "",
                    "sequence_accuracy_improvement_over_random": "",
                }
            )
    return rows


def build_replicate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    trained = [row for row in rows if row["initialization"] == "pretrained"]
    for replicate_id in REPLICATE_IDS:
        for task_count in TASK_COUNTS:
            pool_macros = []
            for pool in ("a", "b"):
                selected = [
                    row
                    for row in trained
                    if row["replicate_id"] == replicate_id
                    and row["model_pool"] == pool
                    and int(row["base_trained_task_count"]) == task_count
                ]
                if len(selected) != 4 or len({row["target_family"] for row in selected}) != 4:
                    raise ValueError("replicate pool macro lacks four balanced targets")
                pool_macros.append(
                    {metric: _mean(float(row[metric]) for row in selected) for metric in (*METRICS, *CONTRASTS)}
                )
            result.append(
                {
                    "replicate_id": replicate_id,
                    "model_seed": int(selected[0]["seed"]),
                    "base_trained_task_count": task_count,
                    "pool_direction_count": 2,
                    "target_count": 8,
                    **{
                        metric: _mean(pool[metric] for pool in pool_macros)
                        for metric in (*METRICS, *CONTRASTS)
                    },
                }
            )
    return result


def build_summary(replicates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for task_count in TASK_COUNTS:
        selected = [row for row in replicates if int(row["base_trained_task_count"]) == task_count]
        if len(selected) != 3:
            raise ValueError("Property32 few-shot summary requires three replicates")
        record: dict[str, Any] = {"base_trained_task_count": task_count, "replicate_count": 3}
        for metric in (*METRICS, *CONTRASTS):
            values = [float(row[metric]) for row in selected]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(record)
    return result


def build_family_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    trained = [row for row in rows if row["initialization"] == "pretrained"]
    result = []
    for task_count in TASK_COUNTS:
        for family in ("local", "positional", "cycle", "global_run"):
            replicate_values = []
            for replicate_id in REPLICATE_IDS:
                selected = [
                    row
                    for row in trained
                    if row["replicate_id"] == replicate_id
                    and int(row["base_trained_task_count"]) == task_count
                    and row["target_family"] == family
                ]
                if len(selected) != 2:
                    raise ValueError("family summary lacks two pool directions")
                replicate_values.append(
                    {metric: _mean(float(row[metric]) for row in selected) for metric in (*METRICS, *CONTRASTS)}
                )
            record: dict[str, Any] = {
                "base_trained_task_count": task_count,
                "target_family": family,
                "replicate_count": 3,
            }
            for metric in (*METRICS, *CONTRASTS):
                values = [item[metric] for item in replicate_values]
                record[f"{metric}_mean"] = statistics.fmean(values)
                record[f"{metric}_sample_sd"] = statistics.stdev(values)
            result.append(record)
    return result


def build_random_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    replicate_values = []
    for replicate_id in REPLICATE_IDS:
        selected = [row for row in rows if row["initialization"] == "random" and row["replicate_id"] == replicate_id]
        if len(selected) != 8:
            raise ValueError("random summary requires eight targets per replicate")
        replicate_values.append({metric: _mean(float(row[metric]) for row in selected) for metric in METRICS})
    record: dict[str, Any] = {"replicate_count": 3}
    for metric in METRICS:
        values = [row[metric] for row in replicate_values]
        record[f"{metric}_mean"] = statistics.fmean(values)
        record[f"{metric}_sample_sd"] = statistics.stdev(values)
    return [record]


def _readme(summary: Sequence[Mapping[str, Any]], random: Mapping[str, Any]) -> str:
    lines = [
        "# Property32 twenty-shot fine-tuning results",
        "",
        "This is Henry Kvinge's fine-tuning notion of generalization on the same ",
        "30 zero-overlap Transformers used for CKA and linear probing. Each model ",
        "is adapted to four balanced opposite-pool properties using 20 support ",
        "examples. Final metrics use 2,500 examples per property from source shard ",
        "199, which was not used by the earlier linear-probe evaluation.",
        "",
        "| k | Adapted exact | Change from zero-shot | Change over random init |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['base_trained_task_count']} | "
            f"{100 * float(row['sequence_accuracy_mean']):.2f}% +/- {100 * float(row['sequence_accuracy_sample_sd']):.2f}% | "
            f"{100 * float(row['sequence_accuracy_improvement_from_zero_shot_mean']):+.2f} +/- {100 * float(row['sequence_accuracy_improvement_from_zero_shot_sample_sd']):.2f} pp | "
            f"{100 * float(row['sequence_accuracy_improvement_over_random_mean']):+.2f} +/- {100 * float(row['sequence_accuracy_improvement_over_random_sample_sd']):.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"The random-initialization control reaches {100 * float(random['sequence_accuracy_mean']):.2f}% +/- "
            f"{100 * float(random['sequence_accuracy_sample_sd']):.2f}% exact accuracy.",
            "",
            "Means first average four target families and both pool directions ",
            "within each replicate, then report mean +/- sample SD across three joint ",
            "task-split/model-seed replicates. The primary contrast is improvement ",
            "from paired zero-shot accuracy; improvement over the seed- and ",
            "support-matched random control is the second confirmatory contrast.",
            "",
            "This experiment measures few-shot adaptability, not hard zero-shot ",
            "execution. See `family_summary.csv` for heterogeneous effects and ",
            "`model_task_results.csv` for every unaveraged adaptation.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def export_results(
    config_path: Path = DEFAULT_CONFIG, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    spec = load_spec(config_path)
    rows = load_rows(config_path)
    replicate_rows = build_replicate_rows(rows)
    summary = build_summary(replicate_rows)
    family = build_family_summary(rows)
    random = build_random_summary(rows)
    artifacts = {
        "model_task_results.csv": (rows, RAW_FIELDS),
        "replicate_summary.csv": (replicate_rows, REPLICATE_FIELDS),
        "summary.csv": (summary, SUMMARY_FIELDS),
        "family_summary.csv": (family, FAMILY_FIELDS),
        "random_summary.csv": (random, RANDOM_FIELDS),
    }
    for name, (values, fields) in artifacts.items():
        _atomic_csv(values, fields, output_dir / name)
    readme = output_dir / "README.md"
    _atomic_text(_readme(summary, random[0]), readme)
    trend = {
        "task_counts": list(TASK_COUNTS),
        "sequence_accuracy": [float(row["sequence_accuracy_mean"]) for row in summary],
        "sequence_accuracy_improvement_from_zero_shot": [
            float(row["sequence_accuracy_improvement_from_zero_shot_mean"]) for row in summary
        ],
        "sequence_accuracy_improvement_over_random": [
            float(row["sequence_accuracy_improvement_over_random_mean"]) for row in summary
        ],
    }
    manifest = {
        "status": "completed",
        "format_version": "property32-fewshot-results/v1",
        "analysis_commit": _git_commit(spec.repository),
        "fewshot_config_sha256": spec.config_sha256,
        "support_artifact_sha256": _sha256(spec.support_artifact),
        "evaluation_manifest_sha256": _sha256(spec.evaluation_dir / "manifest.json"),
        "row_count": len(rows),
        "replicate_count": 3,
        "trend": trend,
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in artifacts},
            "README.md": _sha256(readme),
        },
    }
    if not all(math.isfinite(value) for values in trend.values() if isinstance(values, list) for value in values):
        raise ValueError("Property32 few-shot trend contains non-finite values")
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_results(args.config, args.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_family_summary",
    "build_random_summary",
    "build_replicate_rows",
    "build_summary",
    "export_results",
    "load_rows",
    "main",
]

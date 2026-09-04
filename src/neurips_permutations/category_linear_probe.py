"""Frozen linear probes for the 18 completed v3 category models.

Every probe reads task-free activations at ``<ONE_END>`` and predicts the same
32 scalar permutation properties.  The common target battery makes Encoding
E4, Statistics S4, and Algebra A4 models comparable without feeding a task
token to the base model.  Ridge probes are fitted and tuned on validation data;
the independently selected test sample is evaluated once after this protocol
and implementation have been committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .audit import audit_experiment
from .cka import (
    ActivationSet,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _git_commit,
    _load_random_activations,
    _load_trained_activations,
    _sha256,
    probe_identity,
    select_probe_examples,
)
from .experiments import build_experiment_matrix
from .math_ops import PROPERTY32_TASK_NAMES
from .property_linear_probe import (
    LAYERS,
    METRICS,
    fit_probe_layer,
    property_label_matrix,
    validation_train_tune_indices,
)
from .property_replicates import PROPERTY_FAMILIES


DEFAULT_CONFIG = Path("configs/v3_category_linear_probe.toml")
DEFAULT_OUTPUT_DIR = Path("results/v3/linear-probing/category")
CONDITIONS = ("encoding_e4", "statistics_s4", "algebra_a4")
FAMILY_NAMES = ("local", "positional", "cycle", "global_run")
ARCHITECTURE_LAYERS = {
    "transformer": LAYERS,
    "mlp": ("embedding", "block_01", "final_norm"),
}

RAW_FIELDS = (
    "model_kind",
    "run_id",
    "architecture",
    "condition",
    "model_seed",
    "trained_tasks",
    "probe_task",
    "probe_task_family",
    "layer",
    "selected_ridge_alpha",
    "tuning_length_conditioned_r2",
    "probe_train_examples",
    "probe_tune_examples",
    "probe_test_examples",
    *METRICS,
)

MACRO_FIELDS = (
    "model_kind",
    "run_id",
    "architecture",
    "condition",
    "model_seed",
    "layer",
    "probe_task_family",
    "probe_task_count",
    *METRICS,
)

SUMMARY_FIELDS = (
    "model_kind",
    "architecture",
    "condition",
    "layer",
    "probe_task_family",
    "model_count",
    *tuple(
        field
        for metric in METRICS
        for field in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)

CONTRAST_FIELDS = (
    "architecture",
    "condition",
    "layer",
    "probe_task_family",
    "seed_count",
    *tuple(
        field
        for metric in METRICS
        for field in (f"{metric}_delta_mean", f"{metric}_delta_sample_sd")
    ),
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return tomllib.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def validate_config(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _read_config(path)
    if value.get("protocol_version") != "v3-category-linear-probe/v1":
        raise ValueError("unsupported category linear-probe protocol")
    frozen = value.get("frozen_artifacts")
    probe = value.get("probe")
    if not isinstance(frozen, Mapping) or not isinstance(probe, Mapping):
        raise ValueError("category probe config lacks frozen_artifacts or probe")
    expected_files = {
        "base_experiment_config": "base_experiment_config_sha256",
        "dataset_manifest": "dataset_manifest_sha256",
        "validation_manifest": "validation_manifest_sha256",
        "test_manifest": "test_manifest_sha256",
    }
    for path_key, hash_key in expected_files.items():
        artifact = Path(str(value[path_key]))
        if not artifact.is_file() or _sha256(artifact) != frozen.get(hash_key):
            raise ValueError(f"{path_key} differs from the frozen protocol")
    dataset = json.loads(Path(str(value["dataset_manifest"])).read_text(encoding="utf-8"))
    if dataset.get("schema_version") != frozen.get("dataset_schema_version"):
        raise ValueError("dataset schema differs from the frozen protocol")
    if int(frozen.get("trained_model_count", 0)) != 18:
        raise ValueError("category probe requires exactly 18 trained base models")
    if probe.get("activation_landmark") != "<ONE_END>":
        raise ValueError("category probes must use the task-free <ONE_END> landmark")
    if probe.get("target_registry") != "permutation-properties-32/v1":
        raise ValueError("category probe target registry differs")
    if probe.get("target_transform") != "train-length-conditional-zscore":
        raise ValueError("category probe target transform differs")
    if probe.get("primary_layer") != "final_norm":
        raise ValueError("category probe primary layer differs")
    if probe.get("primary_metric") != "length_conditioned_r2":
        raise ValueError("category probe primary metric differs")
    if probe.get("test_policy") != "single_frozen_pass_after_protocol_and_implementation_commit":
        raise ValueError("category probe test policy differs")
    for key in ("validation_examples", "test_examples"):
        if int(probe.get(key, 0)) < 1_000:
            raise ValueError(f"{key} is too small")
    if int(probe.get("batch_size", 0)) < 1:
        raise ValueError("probe batch_size must be positive")
    fraction = float(probe.get("validation_tune_fraction", 0.0))
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation_tune_fraction must lie in (0, 0.5)")
    alphas = tuple(float(value) for value in probe.get("ridge_alphas", ()))
    if not alphas or tuple(sorted(set(alphas))) != alphas or alphas[0] <= 0:
        raise ValueError("ridge_alphas must be unique positive increasing values")
    seeds = tuple(int(value) for value in probe.get("random_baseline_seeds", ()))
    if seeds != (17, 42, 314159):
        raise ValueError("random baseline seeds differ from the category model seeds")
    if int(probe.get("validation_shard_index", -1)) != 98:
        raise ValueError("validation shard must remain 098")
    if int(probe.get("test_shard_index", -1)) != 99:
        raise ValueError("test shard must remain 099")
    return value, digest


def property_family(task: str) -> str:
    for name, family in zip(FAMILY_NAMES, PROPERTY_FAMILIES, strict=True):
        if task in family:
            return name
    raise ValueError(f"unknown scalar property: {task}")


def _completed_result(output_dir: Path, config_sha256: str) -> dict[str, Any] | None:
    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "completed"
        or value.get("protocol_version") != "v3-category-linear-probe/v1"
        or value.get("config_sha256") != config_sha256
        or value.get("test_split_used") is not True
    ):
        raise ValueError("existing category probe result differs from the frozen protocol")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("existing category probe result lacks an artifact inventory")
    for name, expected in artifacts.items():
        path = output_dir / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"category probe artifact is missing or changed: {name}")
    return dict(value)


def build_macro_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Macro-average task-level probe rows within each model and layer."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        base = tuple(
            row[field]
            for field in (
                "model_kind",
                "run_id",
                "architecture",
                "condition",
                "model_seed",
                "layer",
            )
        )
        groups.setdefault((*base, str(row["probe_task_family"])), []).append(row)
        groups.setdefault((*base, "all"), []).append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        selected = groups[key]
        row = dict(zip(MACRO_FIELDS[:7], key, strict=True))
        row["probe_task_count"] = len(selected)
        for metric in METRICS:
            row[metric] = statistics.fmean(float(item[metric]) for item in selected)
        result.append(row)
    return result


def summarize_macro_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate model-level macros across the three seed-matched models."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "model_kind",
                "architecture",
                "condition",
                "layer",
                "probe_task_family",
            )
        )
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        selected = groups[key]
        seeds = {int(item["model_seed"]) for item in selected}
        if len(selected) != 3 or seeds != {17, 42, 314159}:
            raise ValueError(f"category probe summary lacks three seeds for {key}")
        row = dict(zip(SUMMARY_FIELDS[:5], key, strict=True))
        row["model_count"] = len(selected)
        for metric in METRICS:
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(row)
    return result


def build_contrast_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return trained-minus-random paired-seed contrasts."""

    index = {
        (
            str(row["model_kind"]),
            str(row["architecture"]),
            str(row["condition"]),
            int(row["model_seed"]),
            str(row["layer"]),
            str(row["probe_task_family"]),
        ): row
        for row in rows
    }
    result: list[dict[str, Any]] = []
    for architecture in ("transformer", "mlp"):
        for condition in CONDITIONS:
            for layer in ARCHITECTURE_LAYERS[architecture]:
                for family in (*FAMILY_NAMES, "all"):
                    deltas = {metric: [] for metric in METRICS}
                    for seed in (17, 42, 314159):
                        trained = index[("trained", architecture, condition, seed, layer, family)]
                        random = index[("random", architecture, "random_init", seed, layer, family)]
                        for metric in METRICS:
                            deltas[metric].append(float(trained[metric]) - float(random[metric]))
                    row: dict[str, Any] = {
                        "architecture": architecture,
                        "condition": condition,
                        "layer": layer,
                        "probe_task_family": family,
                        "seed_count": 3,
                    }
                    for metric, values in deltas.items():
                        row[f"{metric}_delta_mean"] = statistics.fmean(values)
                        row[f"{metric}_delta_sample_sd"] = statistics.stdev(values)
                    result.append(row)
    return result


def _run_mapping(run: Any, audit_runs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    audited = audit_runs.get(run.run_id)
    if audited is None or audited.get("status") != "passed":
        raise ValueError(f"base-model audit did not pass: {run.run_id}")
    return {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "seed": run.seed,
        "checkpoint_sha256": str(audited["checkpoint_sha256"]),
        "checkpoint_path": str(audited["checkpoint_path"]),
    }


def _probe_rows(
    validation: ActivationSet,
    test: ActivationSet,
    *,
    validation_labels: Tensor,
    test_labels: Tensor,
    validation_lengths: Tensor,
    test_lengths: Tensor,
    train_indices: Sequence[int],
    tune_indices: Sequence[int],
    alphas: Sequence[float],
    metadata: Mapping[str, Any],
    trained_tasks: Sequence[str],
) -> list[dict[str, Any]]:
    if validation.checkpoint_sha256 != test.checkpoint_sha256:
        raise ValueError("validation and test activations use different checkpoints")
    expected_layers = ARCHITECTURE_LAYERS[str(metadata["architecture"])]
    if tuple(validation.layers) != expected_layers or tuple(test.layers) != expected_layers:
        raise ValueError("category probe layer grid differs from the frozen protocol")
    rows: list[dict[str, Any]] = []
    for layer in expected_layers:
        metrics = fit_probe_layer(
            validation.layers[layer],
            test.layers[layer],
            validation_labels,
            test_labels,
            validation_lengths,
            test_lengths,
            train_indices,
            tune_indices,
            alphas,
        )
        for task, result in zip(PROPERTY32_TASK_NAMES, metrics, strict=True):
            rows.append(
                {
                    **metadata,
                    "trained_tasks": "|".join(trained_tasks),
                    "probe_task": task,
                    "probe_task_family": property_family(task),
                    "layer": layer,
                    "probe_train_examples": len(train_indices),
                    "probe_tune_examples": len(tune_indices),
                    "probe_test_examples": len(test_labels),
                    **result,
                }
            )
    return rows


def _render_readme(
    summary: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    *,
    validation_examples: int,
    test_examples: int,
) -> str:
    final_all = {
        (str(row["model_kind"]), str(row["architecture"]), str(row["condition"])): row
        for row in summary
        if row["layer"] == "final_norm" and row["probe_task_family"] == "all"
    }
    final_contrasts = {
        (str(row["architecture"]), str(row["condition"])): row
        for row in contrasts
        if row["layer"] == "final_norm" and row["probe_task_family"] == "all"
    }
    lines = [
        "# V3 category-model linear probing",
        "",
        "Frozen ridge probes read task-free `<ONE_END>` activations from the 18 ",
        "completed Encoding E4, Statistics S4, and Algebra A4 models. Every model ",
        "is evaluated on the same neutral battery of 32 scalar permutation ",
        "properties. Base-model weights remain frozen.",
        "",
        f"Probes are fitted and tuned on {validation_examples:,} validation ",
        f"permutations and evaluated once on {test_examples:,} independently ",
        "selected test permutations. Targets are standardized within permutation ",
        "length, so the primary R2 measures signal beyond a length-only baseline.",
        "",
        "## Final-layer result",
        "",
        "Values are property-macro means followed by sample SD across three seeds. ",
        "The delta is paired trained minus random initialization at the same ",
        "architecture and seed.",
        "",
        "| Architecture | Training condition | R2 | R2 delta vs random | Exact | Exact delta vs random |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for architecture in ("transformer", "mlp"):
        random = final_all[("random", architecture, "random_init")]
        for condition in CONDITIONS:
            row = final_all[("trained", architecture, condition)]
            delta = final_contrasts[(architecture, condition)]
            lines.append(
                f"| {architecture.title()} | {condition} | "
                f"{float(row['length_conditioned_r2_mean']):.4f} +/- "
                f"{float(row['length_conditioned_r2_sample_sd']):.4f} | "
                f"{float(delta['length_conditioned_r2_delta_mean']):+.4f} +/- "
                f"{float(delta['length_conditioned_r2_delta_sample_sd']):.4f} | "
                f"{100 * float(row['exact_accuracy_mean']):.2f}% +/- "
                f"{100 * float(row['exact_accuracy_sample_sd']):.2f}% | "
                f"{100 * float(delta['exact_accuracy_delta_mean']):+.2f} +/- "
                f"{100 * float(delta['exact_accuracy_delta_sample_sd']):.2f} pp |"
            )
        lines.append(
            f"| {architecture.title()} | random_init | "
            f"{float(random['length_conditioned_r2_mean']):.4f} +/- "
            f"{float(random['length_conditioned_r2_sample_sd']):.4f} | -- | "
            f"{100 * float(random['exact_accuracy_mean']):.2f}% +/- "
            f"{100 * float(random['exact_accuracy_sample_sd']):.2f}% | -- |"
        )
    lines.extend(
        [
            "",
            "The complete CSVs separate local, positional, cycle, and global/run ",
            "target families at every layer. These are linear-decodability results, ",
            "not behavioral task accuracy and not evidence that the base model ",
            "causally uses a decoded property.",
            "",
            "## Artifacts",
            "",
            "- `model_property_layer_probes.csv`: every model/property/layer result;",
            "- `model_family_macro_probes.csv`: property-family macros per model;",
            "- `category_probe_summary.csv`: means and sample SD across seeds;",
            "- `paired_random_contrasts.csv`: seed-paired trained-minus-random controls;",
            "- `probe_manifests.json`, `run_provenance.json`, and `manifest.json`: ",
            "  sample, checkpoint, code, and artifact provenance.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def run_category_linear_probes(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    device: torch.device,
) -> dict[str, Any]:
    config, config_sha256 = validate_config(config_path)
    probe = config["probe"]
    output_dir = output_dir or Path(str(config["output_dir"]))
    completed = _completed_result(output_dir, config_sha256)
    if completed is not None:
        return completed

    repository = Path.cwd()
    base_config = Path(str(config["base_experiment_config"]))
    audit = audit_experiment(base_config, matrix="category")
    if not audit.get("ok") or audit.get("passed_count") != 18:
        raise ValueError("all 18 category base models must pass strict audit")
    audit_runs = {str(row["run_id"]): row for row in audit["runs"]}
    runs = build_experiment_matrix(base_config, matrix="category")
    if len(runs) != 18:
        raise ValueError("category experiment matrix must contain 18 runs")

    dataset_manifest = Path(str(config["dataset_manifest"]))
    validation_examples = select_probe_examples(
        dataset_manifest,
        count=int(probe["validation_examples"]),
        seed=int(probe["validation_probe_seed"]),
        shard_index=int(probe["validation_shard_index"]),
    )
    test_examples = select_probe_examples(
        dataset_manifest,
        count=int(probe["test_examples"]),
        seed=int(probe["test_probe_seed"]),
        shard_index=int(probe["test_shard_index"]),
    )
    validation_identity = probe_identity(
        validation_examples,
        dataset_manifest_sha256=_sha256(dataset_manifest),
        shard_index=int(probe["validation_shard_index"]),
        seed=int(probe["validation_probe_seed"]),
    )
    test_identity = probe_identity(
        test_examples,
        dataset_manifest_sha256=_sha256(dataset_manifest),
        shard_index=int(probe["test_shard_index"]),
        seed=int(probe["test_probe_seed"]),
    )
    test_identity["split"] = "test"
    train_indices, tune_indices = validation_train_tune_indices(
        validation_examples,
        tune_fraction=float(probe["validation_tune_fraction"]),
        seed=int(probe["validation_split_seed"]),
    )
    partition = {
        "train_record_ids": [validation_examples[index].record_id for index in train_indices],
        "tune_record_ids": [validation_examples[index].record_id for index in tune_indices],
    }
    validation_identity["probe_partition_sha256"] = _canonical_sha256(partition)
    validation_identity["probe_train_examples"] = len(train_indices)
    validation_identity["probe_tune_examples"] = len(tune_indices)
    _atomic_json(
        {"validation": validation_identity, "test": test_identity, "partition": partition},
        output_dir / "probe_manifests.json",
    )

    validation_labels = property_label_matrix(validation_examples)
    test_labels = property_label_matrix(test_examples)
    validation_lengths = torch.tensor([row.n for row in validation_examples], dtype=torch.long)
    test_lengths = torch.tensor([row.n for row in test_examples], dtype=torch.long)
    alphas = tuple(float(value) for value in probe["ridge_alphas"])
    batch_size = int(probe["batch_size"])

    raw_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    references: dict[str, dict[str, Any]] = {}
    for run in runs:
        mapping = _run_mapping(run, audit_runs)
        references.setdefault(run.architecture, mapping)
        validation_activation = _load_trained_activations(
            mapping,
            repository=repository,
            examples=validation_examples,
            probe_sha256=str(validation_identity["probe_sha256"]),
            cache_dir=output_dir / "cache" / "validation",
            device=device,
            batch_size=batch_size,
        )
        test_activation = _load_trained_activations(
            mapping,
            repository=repository,
            examples=test_examples,
            probe_sha256=str(test_identity["probe_sha256"]),
            cache_dir=output_dir / "cache" / "test",
            device=device,
            batch_size=batch_size,
        )
        metadata = {
            "model_kind": "trained",
            "run_id": run.run_id,
            "architecture": run.architecture,
            "condition": run.condition,
            "model_seed": run.seed,
        }
        print(f"fitting category probes: {run.run_id}", flush=True)
        raw_rows.extend(
            _probe_rows(
                validation_activation,
                test_activation,
                validation_labels=validation_labels,
                test_labels=test_labels,
                validation_lengths=validation_lengths,
                test_lengths=test_lengths,
                train_indices=train_indices,
                tune_indices=tune_indices,
                alphas=alphas,
                metadata=metadata,
                trained_tasks=run.tasks,
            )
        )
        provenance.append(
            {
                **metadata,
                "trained_tasks": list(run.tasks),
                "checkpoint_sha256": mapping["checkpoint_sha256"],
                "checkpoint_path": mapping["checkpoint_path"],
                "validation_probe_sha256": validation_identity["probe_sha256"],
                "test_probe_sha256": test_identity["probe_sha256"],
            }
        )
        del validation_activation, test_activation

    for architecture in ("transformer", "mlp"):
        reference = references[architecture]
        for seed in (17, 42, 314159):
            validation_activation = _load_random_activations(
                reference,
                repository=repository,
                seed=seed,
                examples=validation_examples,
                probe_sha256=str(validation_identity["probe_sha256"]),
                cache_dir=output_dir / "cache" / f"random-{architecture}-validation",
                device=device,
                batch_size=batch_size,
            )
            test_activation = _load_random_activations(
                reference,
                repository=repository,
                seed=seed,
                examples=test_examples,
                probe_sha256=str(test_identity["probe_sha256"]),
                cache_dir=output_dir / "cache" / f"random-{architecture}-test",
                device=device,
                batch_size=batch_size,
            )
            metadata = {
                "model_kind": "random",
                "run_id": f"random-{architecture}-seed{seed}",
                "architecture": architecture,
                "condition": "random_init",
                "model_seed": seed,
            }
            print(f"fitting category probes: random-{architecture}-seed{seed}", flush=True)
            raw_rows.extend(
                _probe_rows(
                    validation_activation,
                    test_activation,
                    validation_labels=validation_labels,
                    test_labels=test_labels,
                    validation_lengths=validation_lengths,
                    test_lengths=test_lengths,
                    train_indices=train_indices,
                    tune_indices=tune_indices,
                    alphas=alphas,
                    metadata=metadata,
                    trained_tasks=(),
                )
            )
            provenance.append(
                {
                    **metadata,
                    "trained_tasks": [],
                    "checkpoint_sha256": "random_init",
                    "checkpoint_path": None,
                    "validation_probe_sha256": validation_identity["probe_sha256"],
                    "test_probe_sha256": test_identity["probe_sha256"],
                }
            )
            del validation_activation, test_activation

    model_layer_count = 12 * len(ARCHITECTURE_LAYERS["transformer"]) + 12 * len(
        ARCHITECTURE_LAYERS["mlp"]
    )
    expected_rows = model_layer_count * len(PROPERTY32_TASK_NAMES)
    if len(raw_rows) != expected_rows:
        raise ValueError(f"category probe grid has {len(raw_rows)} rows, expected {expected_rows}")
    macro_rows = build_macro_rows(raw_rows)
    summary_rows = summarize_macro_rows(macro_rows)
    contrast_rows = build_contrast_rows(macro_rows)
    for rows in (raw_rows, macro_rows, summary_rows, contrast_rows):
        for row in rows:
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("category probe output contains a non-finite value")

    tables = {
        "model_property_layer_probes.csv": (raw_rows, RAW_FIELDS),
        "model_family_macro_probes.csv": (macro_rows, MACRO_FIELDS),
        "category_probe_summary.csv": (summary_rows, SUMMARY_FIELDS),
        "paired_random_contrasts.csv": (contrast_rows, CONTRAST_FIELDS),
    }
    for name, (rows, fields) in tables.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme_path = output_dir / "README.md"
    _atomic_text(
        _render_readme(
            summary_rows,
            contrast_rows,
            validation_examples=len(validation_examples),
            test_examples=len(test_examples),
        ),
        readme_path,
    )
    provenance_path = output_dir / "run_provenance.json"
    _atomic_json(provenance, provenance_path)
    manifest = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "analysis_commit": _git_commit(repository),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "trained_model_count": 18,
        "random_model_count": 6,
        "property_count": len(PROPERTY32_TASK_NAMES),
        "property_families": list(FAMILY_NAMES),
        "layer_count_by_architecture": {
            name: len(layers) for name, layers in ARCHITECTURE_LAYERS.items()
        },
        "raw_row_count": len(raw_rows),
        "validation_probe": validation_identity,
        "test_probe": test_identity,
        "test_split_used": True,
        "test_use": "single_frozen_category_linear_probe_evaluation",
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in tables},
            "README.md": _sha256(readme_path),
            "probe_manifests.json": _sha256(output_dir / "probe_manifests.json"),
            "run_provenance.json": _sha256(provenance_path),
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    result = run_category_linear_probes(args.config, args.output_dir, device=device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "ARCHITECTURE_LAYERS",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "FAMILY_NAMES",
    "build_contrast_rows",
    "build_macro_rows",
    "main",
    "property_family",
    "run_category_linear_probes",
    "summarize_macro_rows",
    "validate_config",
]

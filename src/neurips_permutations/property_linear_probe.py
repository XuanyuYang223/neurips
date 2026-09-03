"""Frozen linear probes for the 32-property zero-overlap Transformer matrix.

The probe observes task-free activations at ``<ONE_END>`` and predicts all 32
scalar permutation properties. Ridge probes are trained and tuned on a fixed
validation sample, then evaluated once on an independently selected test
sample. Targets are standardized within permutation length using training
statistics, so the primary R2 is measured against a length-only baseline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .cka import (
    ActivationSet,
    ProbeExample,
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
from .math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
from .passage import ID_TO_TOKEN
from .property_experiments import PropertyExperimentRun, build_property_matrix, matrix_summary
from .property_replicates import PROPERTY_FAMILIES, REPLICATE_CONFIGS, validate_replicate_design


DEFAULT_CONFIG = Path("configs/property32_linear_probe.toml")
DEFAULT_OUTPUT_DIR = Path("results/property32-zero-overlap/linear-probing")
LAYERS = ("embedding", "block_01", "block_02", "block_03", "block_04", "final_norm")
METRICS = (
    "length_conditioned_r2",
    "pearson_r",
    "normalized_rmse",
    "exact_accuracy",
    "length_mode_baseline_accuracy",
    "exact_accuracy_minus_baseline",
)

RAW_FIELDS = (
    "model_kind",
    "replicate_id",
    "pool",
    "run_id",
    "model_seed",
    "trained_task_count",
    "trained_tasks",
    "probe_task",
    "probe_task_family",
    "task_status",
    "layer",
    "selected_ridge_alpha",
    "tuning_length_conditioned_r2",
    "probe_train_examples",
    "probe_tune_examples",
    "probe_test_examples",
    *METRICS,
)

RUN_MACRO_FIELDS = (
    "model_kind",
    "replicate_id",
    "pool",
    "run_id",
    "model_seed",
    "trained_task_count",
    "task_status",
    "layer",
    "probe_task_count",
    *METRICS,
)

REPLICATE_FIELDS = (
    "replicate_id",
    "model_seed",
    "trained_task_count",
    "layer",
    "pool_direction_count",
    *METRICS,
)

SUMMARY_FIELDS = (
    "trained_task_count",
    "layer",
    "replicate_count",
    *tuple(
        field
        for metric in METRICS
        for field in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)

RANDOM_FIELDS = (
    "layer",
    "random_seed_count",
    *tuple(
        field
        for metric in METRICS
        for field in (f"{metric}_mean", f"{metric}_sample_sd")
    ),
)


@dataclass(frozen=True)
class TargetNormalizer:
    means: Tensor
    scales: Tensor
    modes: Tensor


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


def _validate_config(path: Path) -> tuple[dict[str, Any], str]:
    value, digest = _read_config(path)
    if value.get("protocol_version") != "property32-linear-probe/v1":
        raise ValueError("unsupported linear-probe protocol")
    if tuple(value.get("replicate_configs", ())) != tuple(
        str(REPLICATE_CONFIGS[key]) for key in ("r0", "r1", "r2")
    ):
        raise ValueError("linear-probe replicate configs differ from the frozen matrix")
    artifact = value.get("data_artifact")
    probe = value.get("probe")
    if not isinstance(artifact, Mapping) or not isinstance(probe, Mapping):
        raise ValueError("linear-probe config lacks artifact or probe settings")
    if artifact.get("schema_version") != "permutation-properties-32/v1":
        raise ValueError("linear-probe dataset schema differs")
    for name in ("validation", "test"):
        manifest = Path(str(value[f"{name}_manifest"]))
        if _sha256(manifest) != artifact.get(f"{name}_manifest_sha256"):
            raise ValueError(f"{name} manifest SHA-256 differs from frozen protocol")
    if probe.get("activation_landmark") != "<ONE_END>":
        raise ValueError("linear probes must use the task-free <ONE_END> landmark")
    if int(probe.get("validation_examples", 0)) < 100:
        raise ValueError("validation probe sample is too small")
    if int(probe.get("test_examples", 0)) < 100:
        raise ValueError("test probe sample is too small")
    fraction = float(probe.get("validation_tune_fraction", 0.0))
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation tune fraction must lie in (0, 0.5)")
    alphas = tuple(float(alpha) for alpha in probe.get("ridge_alphas", ()))
    if not alphas or sorted(set(alphas)) != list(alphas) or alphas[0] <= 0:
        raise ValueError("ridge alphas must be unique positive increasing values")
    if probe.get("primary_layer") != "final_norm":
        raise ValueError("primary probe layer must remain final_norm")
    if probe.get("primary_task_status") != "opposite_pool":
        raise ValueError("primary probe target set must remain opposite_pool")
    validate_replicate_design()
    return value, digest


def _completed_result(output_dir: Path, config_sha256: str) -> dict[str, Any] | None:
    """Return an authenticated completed result without touching test examples."""

    path = output_dir / "manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "completed"
        or value.get("protocol_version") != "property32-linear-probe/v1"
        or value.get("config_sha256") != config_sha256
        or value.get("test_split_used") is not True
    ):
        raise ValueError("existing linear-probe result does not match the frozen protocol")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("existing linear-probe result has no artifact inventory")
    for name, expected_sha256 in artifacts.items():
        artifact = output_dir / str(name)
        if not artifact.is_file() or _sha256(artifact) != expected_sha256:
            raise ValueError(f"existing linear-probe artifact is missing or changed: {name}")
    return dict(value)


def decode_probe_permutation(example: ProbeExample) -> tuple[int, ...]:
    """Decode and validate the canonical one-line prefix in a probe example."""

    tokens = tuple(ID_TO_TOKEN[index] for index in example.token_ids)
    try:
        start = tokens.index("<ONE_START>") + 1
        end = tokens.index("<ONE_END>")
    except ValueError as error:
        raise ValueError("probe does not contain canonical one-line boundaries") from error
    payload = tokens[start:end]
    values = tuple(int(token) for token in payload if token != ",")
    expected = tuple(range(1, example.n + 1))
    if len(values) != example.n or tuple(sorted(values)) != expected:
        raise ValueError("probe one-line payload is not a permutation of 1,...,n")
    return values


def property_label_matrix(examples: Sequence[ProbeExample]) -> Tensor:
    """Recompute all 32 labels from task-free permutation inputs."""

    labels = [
        [PROPERTY_FUNCTIONS[task](decode_probe_permutation(example)) for task in PROPERTY32_TASK_NAMES]
        for example in examples
    ]
    result = torch.tensor(labels, dtype=torch.float64)
    if result.shape != (len(examples), len(PROPERTY32_TASK_NAMES)):
        raise ValueError("property label matrix has the wrong shape")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("property label matrix contains non-finite values")
    return result


def validation_train_tune_indices(
    examples: Sequence[ProbeExample], *, tune_fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a deterministic length-stratified train/tune partition."""

    by_length: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        by_length.setdefault(example.n, []).append(index)
    train: list[int] = []
    tune: list[int] = []
    for n, indices in sorted(by_length.items()):
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}:{examples[index].record_id}".encode("ascii")
            ).digest(),
        )
        tune_count = max(1, min(len(ranked) - 1, round(len(ranked) * tune_fraction)))
        tune.extend(ranked[:tune_count])
        train.extend(ranked[tune_count:])
        if not train or not tune:  # pragma: no cover - guarded per stratum above
            raise ValueError(f"length {n} could not be split")
    train_result = tuple(sorted(train))
    tune_result = tuple(sorted(tune))
    if set(train_result) & set(tune_result) or len(train_result) + len(tune_result) != len(examples):
        raise ValueError("validation probe partition is not exhaustive and disjoint")
    return train_result, tune_result


def fit_target_normalizer(labels: Tensor, lengths: Tensor) -> TargetNormalizer:
    """Fit length-conditional target means, scales, and modal labels."""

    if labels.ndim != 2 or lengths.ndim != 1 or labels.shape[0] != lengths.shape[0]:
        raise ValueError("labels and lengths have incompatible shapes")
    maximum = int(lengths.max())
    task_count = labels.shape[1]
    means = torch.zeros((maximum + 1, task_count), dtype=torch.float64)
    scales = torch.ones_like(means)
    modes = torch.zeros_like(means)
    for n in sorted(set(int(value) for value in lengths.tolist())):
        selected = labels[lengths == n]
        means[n] = selected.mean(dim=0)
        scale = torch.sqrt(((selected - means[n]) ** 2).mean(dim=0))
        scales[n] = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        for task_index in range(task_count):
            values, counts = torch.unique(selected[:, task_index], return_counts=True)
            maximum_count = counts.max()
            modes[n, task_index] = values[counts == maximum_count].min()
    return TargetNormalizer(means=means, scales=scales, modes=modes)


def _transform_targets(labels: Tensor, lengths: Tensor, normalizer: TargetNormalizer) -> Tensor:
    if int(lengths.max()) >= normalizer.means.shape[0]:
        raise ValueError("target normalizer does not cover every permutation length")
    return (labels - normalizer.means[lengths]) / normalizer.scales[lengths]


def _standardize_features(train: Tensor, other: Tensor) -> tuple[Tensor, Tensor]:
    mean = train.mean(dim=0)
    scale = torch.sqrt(((train - mean) ** 2).mean(dim=0))
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    return (train - mean) / scale, (other - mean) / scale


def _ridge_predictions(
    train_x: Tensor,
    train_y: Tensor,
    evaluation_x: Tensor,
    alphas: Sequence[float],
) -> tuple[Tensor, ...]:
    """Fit multi-target ridge regressions and predict for every alpha."""

    count = train_x.shape[0]
    covariance = train_x.T @ train_x / count
    cross = train_x.T @ train_y / count
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    projected = eigenvectors.T @ cross
    predictions: list[Tensor] = []
    for alpha in alphas:
        weights = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        predictions.append(evaluation_x @ weights)
    return tuple(predictions)


def _column_r2(truth: Tensor, prediction: Tensor) -> Tensor:
    denominator = truth.square().sum(dim=0)
    if bool((denominator <= 1e-12).any()):
        raise ValueError("a probe target has no conditional variation")
    return 1.0 - (truth - prediction).square().sum(dim=0) / denominator


def _column_pearson(truth: Tensor, prediction: Tensor) -> Tensor:
    centered_truth = truth - truth.mean(dim=0)
    centered_prediction = prediction - prediction.mean(dim=0)
    denominator = torch.sqrt(
        centered_truth.square().sum(dim=0) * centered_prediction.square().sum(dim=0)
    )
    numerator = (centered_truth * centered_prediction).sum(dim=0)
    return torch.where(denominator > 1e-12, numerator / denominator, torch.zeros_like(numerator))


def fit_probe_layer(
    validation_x: Tensor,
    test_x: Tensor,
    validation_labels: Tensor,
    test_labels: Tensor,
    validation_lengths: Tensor,
    test_lengths: Tensor,
    train_indices: Sequence[int],
    tune_indices: Sequence[int],
    alphas: Sequence[float],
) -> list[dict[str, float]]:
    """Select ridge coefficients on tuning data and evaluate once on test data."""

    tensors = (validation_x, test_x, validation_labels, test_labels)
    if not all(tensor.ndim == 2 and bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("probe inputs must be finite matrices")
    validation_x = validation_x.to(dtype=torch.float64)
    test_x = test_x.to(dtype=torch.float64)
    train_index = torch.tensor(train_indices, dtype=torch.long)
    tune_index = torch.tensor(tune_indices, dtype=torch.long)

    selection_normalizer = fit_target_normalizer(
        validation_labels[train_index], validation_lengths[train_index]
    )
    train_y = _transform_targets(
        validation_labels[train_index], validation_lengths[train_index], selection_normalizer
    )
    tune_y = _transform_targets(
        validation_labels[tune_index], validation_lengths[tune_index], selection_normalizer
    )
    train_x, tune_x = _standardize_features(
        validation_x[train_index], validation_x[tune_index]
    )
    tune_predictions = _ridge_predictions(train_x, train_y, tune_x, alphas)
    tuning_scores = torch.stack([_column_r2(tune_y, value) for value in tune_predictions])
    best_indices = tuning_scores.argmax(dim=0)

    final_normalizer = fit_target_normalizer(validation_labels, validation_lengths)
    final_y = _transform_targets(validation_labels, validation_lengths, final_normalizer)
    test_y = _transform_targets(test_labels, test_lengths, final_normalizer)
    final_x, standardized_test_x = _standardize_features(validation_x, test_x)
    test_predictions = _ridge_predictions(final_x, final_y, standardized_test_x, alphas)
    prediction = torch.empty_like(test_y)
    for task_index in range(test_y.shape[1]):
        prediction[:, task_index] = test_predictions[int(best_indices[task_index])][:, task_index]

    r2 = _column_r2(test_y, prediction)
    pearson = _column_pearson(test_y, prediction)
    rmse = torch.sqrt((test_y - prediction).square().mean(dim=0))
    raw_prediction = (
        prediction * final_normalizer.scales[test_lengths]
        + final_normalizer.means[test_lengths]
    )
    exact = torch.round(raw_prediction).eq(test_labels).double().mean(dim=0)
    baseline = final_normalizer.modes[test_lengths].eq(test_labels).double().mean(dim=0)

    results: list[dict[str, float]] = []
    for task_index in range(test_y.shape[1]):
        selected = int(best_indices[task_index])
        result = {
            "selected_ridge_alpha": float(alphas[selected]),
            "tuning_length_conditioned_r2": float(tuning_scores[selected, task_index]),
            "length_conditioned_r2": float(r2[task_index]),
            "pearson_r": float(pearson[task_index]),
            "normalized_rmse": float(rmse[task_index]),
            "exact_accuracy": float(exact[task_index]),
            "length_mode_baseline_accuracy": float(baseline[task_index]),
            "exact_accuracy_minus_baseline": float(exact[task_index] - baseline[task_index]),
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("linear-probe result is not finite")
        results.append(result)
    return results


def _property_family(task: str) -> str:
    for index, family in enumerate(PROPERTY_FAMILIES):
        if task in family:
            return ("local", "positional", "cycle", "global_run")[index]
    raise ValueError(f"unknown property family: {task}")


def _run_mapping(run: PropertyExperimentRun) -> dict[str, Any]:
    marker_path = Path(run.output_dir) / "completed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(marker["checkpoint"]))
    if not checkpoint.is_file():
        checkpoint = marker_path.parent / checkpoint.name
    if _sha256(checkpoint) != marker.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint hash mismatch: {run.run_id}")
    return {
        "run_id": run.run_id,
        "architecture": run.architecture,
        "task_count": run.task_count,
        "seed": run.seed,
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "checkpoint_path": str(checkpoint),
    }


def _task_status(task: str, run: PropertyExperimentRun, pool_tasks: Sequence[str]) -> str:
    if task in run.tasks:
        return "trained"
    if task in pool_tasks:
        return "same_pool_untrained"
    return "opposite_pool"


def build_run_macro_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "model_kind",
                "replicate_id",
                "pool",
                "run_id",
                "model_seed",
                "trained_task_count",
                "task_status",
                "layer",
            )
        )
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        selected = groups[key]
        value = dict(zip(RUN_MACRO_FIELDS[:8], key))
        value["probe_task_count"] = len(selected)
        for metric in METRICS:
            value[metric] = statistics.fmean(float(row[metric]) for row in selected)
        result.append(value)
    return result


def build_primary_replicate_rows(
    run_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replicate_ids = tuple(REPLICATE_CONFIGS)
    for replicate_id in replicate_ids:
        for task_count in (1, 2, 4, 8, 16):
            for layer in LAYERS:
                selected = [
                    row
                    for row in run_rows
                    if row["model_kind"] == "trained"
                    and row["replicate_id"] == replicate_id
                    and int(row["trained_task_count"]) == task_count
                    and row["task_status"] == "opposite_pool"
                    and row["layer"] == layer
                ]
                if len(selected) != 2 or {row["pool"] for row in selected} != {"a", "b"}:
                    raise ValueError("primary probe grid lacks both pool directions")
                row: dict[str, Any] = {
                    "replicate_id": replicate_id,
                    "model_seed": int(selected[0]["model_seed"]),
                    "trained_task_count": task_count,
                    "layer": layer,
                    "pool_direction_count": 2,
                }
                for metric in METRICS:
                    row[metric] = statistics.fmean(float(item[metric]) for item in selected)
                result.append(row)
    return result


def summarize_primary_rows(
    replicate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for task_count in (1, 2, 4, 8, 16):
        for layer in LAYERS:
            selected = [
                row
                for row in replicate_rows
                if int(row["trained_task_count"]) == task_count and row["layer"] == layer
            ]
            if len(selected) != 3 or {row["replicate_id"] for row in selected} != set(REPLICATE_CONFIGS):
                raise ValueError("primary probe summary lacks three replicates")
            row: dict[str, Any] = {
                "trained_task_count": task_count,
                "layer": layer,
                "replicate_count": len(selected),
            }
            for metric in METRICS:
                values = [float(item[metric]) for item in selected]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_sample_sd"] = statistics.stdev(values)
            result.append(row)
    return result


def summarize_random_rows(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in LAYERS:
        selected = [
            row
            for row in run_rows
            if row["model_kind"] == "random" and row["layer"] == layer
        ]
        if len(selected) != 3:
            raise ValueError("random probe summary requires three seeds")
        row: dict[str, Any] = {"layer": layer, "random_seed_count": 3}
        for metric in METRICS:
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values)
        result.append(row)
    return result


def _rank(values: Sequence[float]) -> list[float]:
    """Return average ranks, including the mathematically correct tie handling."""

    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def _primary_trend(summary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = sorted(
        (row for row in summary if row["layer"] == "final_norm"),
        key=lambda row: int(row["trained_task_count"]),
    )
    task_counts = [int(row["trained_task_count"]) for row in selected]
    values = [float(row["length_conditioned_r2_mean"]) for row in selected]
    exact = [float(row["exact_accuracy_mean"]) for row in selected]
    if task_counts != [1, 2, 4, 8, 16]:
        raise ValueError("primary linear-probe trend has the wrong k grid")
    return {
        "task_counts": task_counts,
        "final_layer_opposite_pool_length_conditioned_r2": values,
        "final_layer_opposite_pool_exact_accuracy": exact,
        "spearman_rho_r2_vs_k": _pearson(_rank(task_counts), _rank(values)),
        "monotonic_non_decreasing_r2": all(a <= b for a, b in zip(values, values[1:])),
        "delta_k16_minus_k1_r2": values[-1] - values[0],
        "best_k_by_r2": task_counts[max(range(len(values)), key=values.__getitem__)],
    }


def _render_readme(
    summary: Sequence[Mapping[str, Any]],
    random_summary: Sequence[Mapping[str, Any]],
    trend: Mapping[str, Any],
    *,
    validation_examples: int,
    test_examples: int,
) -> str:
    final = {
        int(row["trained_task_count"]): row
        for row in summary
        if row["layer"] == "final_norm"
    }
    random_final = next(row for row in random_summary if row["layer"] == "final_norm")
    lines = [
        "# Permutation linear-probing results",
        "",
        "Linear ridge probes read task-free `<ONE_END>` activations from the ",
        "30 completed zero-overlap Transformers. Probes were trained and tuned ",
        f"on {validation_examples:,} validation permutations and evaluated once ",
        f"on {test_examples:,} independent test permutations.",
        "",
        "## Opposite-pool final-layer results",
        "",
        "Each value first macro-averages the 16 unseen opposite-pool properties ",
        "within a model, then the two pool directions within a replicate, and ",
        "finally reports mean +/- sample SD across three joint task-split/model-seed ",
        "replicates.",
        "",
        "| k | Length-conditioned R2 | Exact accuracy | Length-mode baseline | Exact minus baseline |",
        "|---:|---:|---:|---:|---:|",
    ]
    for task_count in (1, 2, 4, 8, 16):
        row = final[task_count]
        lines.append(
            f"| {task_count} | {float(row['length_conditioned_r2_mean']):.4f} +/- "
            f"{float(row['length_conditioned_r2_sample_sd']):.4f} | "
            f"{100 * float(row['exact_accuracy_mean']):.2f}% +/- "
            f"{100 * float(row['exact_accuracy_sample_sd']):.2f}% | "
            f"{100 * float(row['length_mode_baseline_accuracy_mean']):.2f}% | "
            f"{100 * float(row['exact_accuracy_minus_baseline_mean']):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"Final-layer R2 Spearman rho across k: "
            f"{float(trend['spearman_rho_r2_vs_k']):.4f}.",
            f"k=16 minus k=1 R2: {float(trend['delta_k16_minus_k1_r2']):+.4f}.",
            f"Best mean R2 occurs at k={trend['best_k_by_r2']}; monotonic "
            f"non-decreasing: {str(bool(trend['monotonic_non_decreasing_r2'])).lower()}.",
            "",
            "## Random-initialization control",
            "",
            f"Final-layer length-conditioned R2: "
            f"{float(random_final['length_conditioned_r2_mean']):.4f} +/- "
            f"{float(random_final['length_conditioned_r2_sample_sd']):.4f}.",
            f"Final-layer exact accuracy: "
            f"{100 * float(random_final['exact_accuracy_mean']):.2f}% +/- "
            f"{100 * float(random_final['exact_accuracy_sample_sd']):.2f}%.",
            "",
            "R2 is measured after removing the training-set conditional mean and ",
            "scale within each permutation length. Exact accuracy rounds the linear ",
            "prediction back to an integer property value. The raw CSV retains every ",
            "model, target property, layer, selected ridge coefficient, and metric.",
            "",
            "This is a linear decodability result, not evidence that the base model ",
            "can behaviorally execute an unseen operation or that the decoded feature ",
            "is causally used by the model.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def _probe_rows_for_activation_pair(
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
    pool_tasks: Sequence[str],
) -> list[dict[str, Any]]:
    if validation.checkpoint_sha256 != test.checkpoint_sha256:
        raise ValueError("validation and test activations use different checkpoints")
    if tuple(validation.layers) != LAYERS or tuple(test.layers) != LAYERS:
        raise ValueError("linear-probe layer grid differs from the frozen protocol")
    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
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
        for task, result in zip(PROPERTY32_TASK_NAMES, metrics):
            if metadata["model_kind"] == "random":
                status = "random_baseline"
            elif task in trained_tasks:
                status = "trained"
            elif task in pool_tasks:
                status = "same_pool_untrained"
            else:
                status = "opposite_pool"
            rows.append(
                {
                    **metadata,
                    "trained_tasks": "|".join(trained_tasks),
                    "probe_task": task,
                    "probe_task_family": _property_family(task),
                    "task_status": status,
                    "layer": layer,
                    "probe_train_examples": len(train_indices),
                    "probe_tune_examples": len(tune_indices),
                    "probe_test_examples": len(test_labels),
                    **result,
                }
            )
    return rows


def run_linear_probes(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    *,
    device: torch.device,
) -> dict[str, Any]:
    config, config_sha256 = _validate_config(config_path)
    probe_config = config["probe"]
    output_dir = output_dir or Path(str(config["output_dir"]))
    completed = _completed_result(output_dir, config_sha256)
    if completed is not None:
        return completed
    repository = Path.cwd()
    validation_manifest = Path(str(config["validation_manifest"]))
    test_manifest = Path(str(config["test_manifest"]))

    validation_examples = select_probe_examples(
        validation_manifest,
        count=int(probe_config["validation_examples"]),
        seed=int(probe_config["validation_probe_seed"]),
        shard_index=0,
    )
    test_examples = select_probe_examples(
        test_manifest,
        count=int(probe_config["test_examples"]),
        seed=int(probe_config["test_probe_seed"]),
        shard_index=0,
    )
    validation_probe = probe_identity(
        validation_examples,
        dataset_manifest_sha256=_sha256(validation_manifest),
        shard_index=0,
        seed=int(probe_config["validation_probe_seed"]),
    )
    validation_probe["split"] = "validation"
    test_probe = probe_identity(
        test_examples,
        dataset_manifest_sha256=_sha256(test_manifest),
        shard_index=0,
        seed=int(probe_config["test_probe_seed"]),
    )
    test_probe["split"] = "test"
    train_indices, tune_indices = validation_train_tune_indices(
        validation_examples,
        tune_fraction=float(probe_config["validation_tune_fraction"]),
        seed=int(probe_config["validation_split_seed"]),
    )
    split_identity = {
        "train_record_ids": [validation_examples[index].record_id for index in train_indices],
        "tune_record_ids": [validation_examples[index].record_id for index in tune_indices],
    }
    validation_probe["probe_partition_sha256"] = _canonical_sha256(split_identity)
    validation_probe["probe_train_examples"] = len(train_indices)
    validation_probe["probe_tune_examples"] = len(tune_indices)
    _atomic_json(
        {"validation": validation_probe, "test": test_probe, "partition": split_identity},
        output_dir / "probe_manifests.json",
    )

    validation_labels = property_label_matrix(validation_examples)
    test_labels = property_label_matrix(test_examples)
    validation_lengths = torch.tensor([example.n for example in validation_examples], dtype=torch.long)
    test_lengths = torch.tensor([example.n for example in test_examples], dtype=torch.long)
    alphas = tuple(float(alpha) for alpha in probe_config["ridge_alphas"])
    batch_size = int(probe_config["batch_size"])

    raw_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    reference_mapping: dict[str, Any] | None = None
    for replicate_id, replicate_config in REPLICATE_CONFIGS.items():
        status = matrix_summary(replicate_config)
        if status["complete_count"] != status["run_count"]:
            raise ValueError(f"{replicate_id} base-model matrix is incomplete")
        replicate_value, _ = _read_config(replicate_config)
        pool_by_name = {
            "a": tuple(replicate_value["pool_a"]),
            "b": tuple(replicate_value["pool_b"]),
        }
        runs = build_property_matrix(replicate_config)
        validation_cache = (
            Path("results/property32-zero-overlap/replicates") / replicate_id / "cka" / "cache"
        )
        for run in runs:
            mapping = _run_mapping(run)
            reference_mapping = reference_mapping or mapping
            validation_activation = _load_trained_activations(
                mapping,
                repository=repository,
                examples=validation_examples,
                probe_sha256=validation_probe["probe_sha256"],
                cache_dir=validation_cache,
                device=device,
                batch_size=batch_size,
            )
            test_activation = _load_trained_activations(
                mapping,
                repository=repository,
                examples=test_examples,
                probe_sha256=test_probe["probe_sha256"],
                cache_dir=output_dir / "cache" / "test",
                device=device,
                batch_size=batch_size,
            )
            metadata = {
                "model_kind": "trained",
                "replicate_id": replicate_id,
                "pool": run.pool,
                "run_id": run.run_id,
                "model_seed": run.seed,
                "trained_task_count": run.task_count,
            }
            print(f"fitting probes: {run.run_id}")
            raw_rows.extend(
                _probe_rows_for_activation_pair(
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
                    pool_tasks=pool_by_name[run.pool],
                )
            )
            provenance.append(
                {
                    **metadata,
                    "trained_tasks": list(run.tasks),
                    "checkpoint_sha256": mapping["checkpoint_sha256"],
                    "checkpoint_path": mapping["checkpoint_path"],
                    "validation_probe_sha256": validation_probe["probe_sha256"],
                    "test_probe_sha256": test_probe["probe_sha256"],
                }
            )
            del validation_activation, test_activation

    if reference_mapping is None:  # pragma: no cover
        raise ValueError("no reference model is available for random probes")
    for seed in tuple(int(value) for value in probe_config["random_baseline_seeds"]):
        validation_activation = _load_random_activations(
            reference_mapping,
            repository=repository,
            seed=seed,
            examples=validation_examples,
            probe_sha256=validation_probe["probe_sha256"],
            cache_dir=output_dir / "cache" / "random-validation",
            device=device,
            batch_size=batch_size,
        )
        test_activation = _load_random_activations(
            reference_mapping,
            repository=repository,
            seed=seed,
            examples=test_examples,
            probe_sha256=test_probe["probe_sha256"],
            cache_dir=output_dir / "cache" / "random-test",
            device=device,
            batch_size=batch_size,
        )
        metadata = {
            "model_kind": "random",
            "replicate_id": "random",
            "pool": "random",
            "run_id": f"random-transformer-seed{seed}",
            "model_seed": seed,
            "trained_task_count": 0,
        }
        print(f"fitting probes: random-transformer-seed{seed}")
        raw_rows.extend(
            _probe_rows_for_activation_pair(
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
                pool_tasks=(),
            )
        )
        provenance.append(
            {
                **metadata,
                "trained_tasks": [],
                "checkpoint_sha256": "random_init",
                "checkpoint_path": None,
                "validation_probe_sha256": validation_probe["probe_sha256"],
                "test_probe_sha256": test_probe["probe_sha256"],
            }
        )

    expected_raw = (30 + 3) * len(PROPERTY32_TASK_NAMES) * len(LAYERS)
    if len(raw_rows) != expected_raw:
        raise ValueError(f"linear-probe raw grid has {len(raw_rows)} rows, expected {expected_raw}")
    run_rows = build_run_macro_rows(raw_rows)
    replicate_rows = build_primary_replicate_rows(run_rows)
    summary_rows = summarize_primary_rows(replicate_rows)
    random_rows = summarize_random_rows(run_rows)
    trend = _primary_trend(summary_rows)

    artifact_rows = {
        "model_task_layer_probes.csv": (raw_rows, RAW_FIELDS),
        "model_macro_probes.csv": (run_rows, RUN_MACRO_FIELDS),
        "opposite_pool_replicates.csv": (replicate_rows, REPLICATE_FIELDS),
        "opposite_pool_summary.csv": (summary_rows, SUMMARY_FIELDS),
        "random_baseline_summary.csv": (random_rows, RANDOM_FIELDS),
    }
    for name, (rows, fields) in artifact_rows.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme = output_dir / "README.md"
    _atomic_text(
        _render_readme(
            summary_rows,
            random_rows,
            trend,
            validation_examples=len(validation_examples),
            test_examples=len(test_examples),
        ),
        readme,
    )
    provenance_path = output_dir / "run_provenance.json"
    _atomic_json(provenance, provenance_path)
    manifest = {
        "status": "completed",
        "protocol_version": config["protocol_version"],
        "analysis_commit": _git_commit(repository),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "trained_model_count": 30,
        "random_model_count": 3,
        "property_count": len(PROPERTY32_TASK_NAMES),
        "layer_count": len(LAYERS),
        "raw_row_count": len(raw_rows),
        "validation_probe": validation_probe,
        "test_probe": test_probe,
        "primary_trend": trend,
        "test_split_used": True,
        "test_use": "single_frozen_linear-probe_evaluation",
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in artifact_rows},
            "README.md": _sha256(readme),
            "probe_manifests.json": _sha256(output_dir / "probe_manifests.json"),
            provenance_path.name: _sha256(provenance_path),
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
    result = run_linear_probes(args.config, args.output_dir, device=device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "TargetNormalizer",
    "build_primary_replicate_rows",
    "build_run_macro_rows",
    "decode_probe_permutation",
    "fit_probe_layer",
    "fit_target_normalizer",
    "main",
    "property_label_matrix",
    "run_linear_probes",
    "summarize_primary_rows",
    "validation_train_tune_indices",
]

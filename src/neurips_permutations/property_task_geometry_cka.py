"""Preregistered CKA analyses for the combinatorial task-geometry study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations, product
import json
import math
from pathlib import Path
import random
import statistics
import tomllib
from typing import Any, Mapping, Sequence

import torch

from .cka import (
    ActivationSet,
    ProbeExample,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _canonical_sha256,
    _git_commit,
    _load_random_activations,
    _load_trained_activations,
    _sha256,
    linear_cka,
    probe_identity,
    select_probe_examples,
)
from .passage import ID_TO_TOKEN, TOKEN_TO_ID, encode_number, one_line_tokens
from .property_task_geometry_experiments import (
    BUNDLE_SPLIT_IDS,
    DEFAULT_CONFIG,
    MODEL_SEEDS,
    RELATED_PAIR_COUNTS,
    GeometryRun,
    TaskRelation,
    build_geometry_matrix,
    geometry_summary,
    relation_definitions,
    transform_permutation,
)


DEFAULT_OUTPUT_DIR = Path("results/property-task-geometry/cka")
SIGNATURE_FIELDS = (
    "architecture",
    "d_model",
    "num_layers",
    "num_heads",
    "dropout",
    "mlp_ratio",
    "tie_embeddings",
    "manifest",
    "validation_manifest",
    "max_steps",
    "batch_size",
    "gradient_accumulation_steps",
    "max_seq_len",
    "max_tokens_per_batch",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "min_lr_ratio",
    "max_grad_norm",
    "shuffle_buffer_size",
    "checkpoint_every",
    "validate_every",
    "validation_batches_per_task",
    "amp",
    "bf16",
)


def _sample_sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires matched sequences of length >= 2")
    xm = statistics.fmean(xs)
    ym = statistics.fmean(ys)
    numerator = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def _decode_probe_permutation(example: ProbeExample) -> tuple[int, ...]:
    tokens = tuple(ID_TO_TOKEN[token] for token in example.token_ids)
    try:
        start = tokens.index("<ONE_START>") + 1
        end = tokens.index("<ONE_END>")
    except ValueError as error:
        raise ValueError("probe does not contain a one-line permutation") from error
    values = tuple(int(token) for token in tokens[start:end] if token != ",")
    if len(values) != example.n or sorted(values) != list(range(1, example.n + 1)):
        raise ValueError("probe contains an invalid one-line permutation")
    return values


def transform_probe_examples(
    examples: Sequence[ProbeExample], transformation: str
) -> tuple[ProbeExample, ...]:
    transformed: list[ProbeExample] = []
    for example in examples:
        value = transform_permutation(
            _decode_probe_permutation(example), transformation
        )
        tokens = (
            "<BOS>",
            "<SIZE>",
            *encode_number(len(value)),
            *one_line_tokens(value),
        )
        transformed.append(
            ProbeExample(
                record_id=example.record_id,
                n=example.n,
                token_ids=tuple(TOKEN_TO_ID[token] for token in tokens),
            )
        )
    return tuple(transformed)


def transformed_probe_identity(
    examples: Sequence[ProbeExample],
    *,
    transformation: str,
    dataset_manifest_sha256: str,
    shard_index: int,
    seed: int,
) -> dict[str, Any]:
    value = probe_identity(
        examples,
        dataset_manifest_sha256=dataset_manifest_sha256,
        shard_index=shard_index,
        seed=seed,
    )
    value["input_transformation"] = transformation
    value["probe_sha256"] = _canonical_sha256(
        {"transformation": transformation, "examples": [asdict(item) for item in examples]}
    )
    return value


def summarize_bundle_rows(
    rows: Sequence[Mapping[str, Any]], *, layer: str = "final_norm"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final = [row for row in rows if str(row["layer"]) == layer]
    expected = {
        (split_id, seed, count)
        for split_id in BUNDLE_SPLIT_IDS
        for seed in MODEL_SEEDS
        for count in RELATED_PAIR_COUNTS
    }
    observed = {
        (str(row["split_id"]), int(row["model_seed"]), int(row["related_pair_count"]))
        for row in final
    }
    if observed != expected or len(final) != 48:
        raise ValueError("bundle CKA grid is incomplete")
    summary: list[dict[str, Any]] = []
    for count in RELATED_PAIR_COUNTS:
        values = [
            float(row["linear_cka"])
            for row in final
            if int(row["related_pair_count"]) == count
        ]
        summary.append(
            {
                "related_pair_count": count,
                "cell_count": len(values),
                "final_layer_cka_mean": statistics.fmean(values),
                "final_layer_cka_sample_sd": _sample_sd(values),
                "final_layer_cka_min": min(values),
                "final_layer_cka_max": max(values),
            }
        )
    curves: dict[str, Any] = {}
    deltas: list[float] = []
    for split_id in BUNDLE_SPLIT_IDS:
        for seed in MODEL_SEEDS:
            selected = sorted(
                (
                    row
                    for row in final
                    if row["split_id"] == split_id and int(row["model_seed"]) == seed
                ),
                key=lambda row: RELATED_PAIR_COUNTS.index(int(row["related_pair_count"])),
            )
            values = [float(row["linear_cka"]) for row in selected]
            delta = values[-1] - values[0]
            deltas.append(delta)
            curves[f"{split_id}:{seed}"] = {
                "values": values,
                "delta_r4_minus_r0": delta,
                "spearman_rho": _pearson(
                    list(map(float, RELATED_PAIR_COUNTS)), _ranks(values)
                ),
                "monotonic_non_decreasing": all(
                    left <= right for left, right in zip(values, values[1:])
                ),
            }
    positives = sum(delta > 0 for delta in deltas)
    tail = sum(
        math.comb(len(deltas), index)
        for index in range(max(positives, len(deltas) - positives), len(deltas) + 1)
    )
    return summary, {
        "cell_curves": curves,
        "delta_r4_minus_r0_mean": statistics.fmean(deltas),
        "delta_r4_minus_r0_sample_sd": _sample_sd(deltas),
        "positive_delta_cells": positives,
        "two_sided_exact_sign_test_p": min(1.0, 2 * tail / (2 ** len(deltas))),
        "monotonic_cell_count": sum(
            value["monotonic_non_decreasing"] for value in curves.values()
        ),
    }


def summarize_specialist_rows(
    rows: Sequence[Mapping[str, Any]], *, layer: str = "final_norm"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final = [row for row in rows if str(row["layer"]) == layer]
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in final:
        if bool(row["same_seed"]):
            continue
        key = (str(row["comparison"]), str(row["task_a"]), str(row["task_b"]))
        grouped.setdefault(key, []).append(float(row["linear_cka"]))
    pair_rows = [
        {
            "comparison": key[0],
            "task_a": key[1],
            "task_b": key[2],
            "seed_comparison_count": len(values),
            "final_layer_cka_mean": statistics.fmean(values),
            "final_layer_cka_sample_sd": _sample_sd(values),
        }
        for key, values in sorted(grouped.items())
    ]
    group_rows: list[dict[str, Any]] = []
    for comparison in ("same_task", "direct_relation", "no_direct_relation"):
        values = [
            float(row["final_layer_cka_mean"])
            for row in pair_rows
            if row["comparison"] == comparison
        ]
        if not values:
            raise ValueError(f"specialist comparison {comparison} is missing")
        group_rows.append(
            {
                "comparison": comparison,
                "task_pair_count": len(values),
                "final_layer_cka_mean": statistics.fmean(values),
                "final_layer_cka_sample_sd": _sample_sd(values),
                "final_layer_cka_min": min(values),
                "final_layer_cka_max": max(values),
            }
        )
    return pair_rows, group_rows


def task_label_permutation_test(
    pair_rows: Sequence[Mapping[str, Any]],
    relations: Sequence[TaskRelation],
    *,
    permutations_count: int,
    seed: int,
) -> dict[str, Any]:
    if permutations_count < 1:
        raise ValueError("permutations_count must be positive")
    tasks = tuple(task for relation in relations for task in (relation.left, relation.right))
    cka = {
        frozenset((str(row["task_a"]), str(row["task_b"]))): float(
            row["final_layer_cka_mean"]
        )
        for row in pair_rows
        if row["comparison"] != "same_task"
    }
    all_pairs = [frozenset(pair) for pair in combinations(tasks, 2)]
    if set(cka) != set(all_pairs):
        raise ValueError("task-pair CKA matrix is incomplete")
    relation_edges = [frozenset((relation.left, relation.right)) for relation in relations]

    def statistic(edges: Sequence[frozenset[str]]) -> float:
        selected = set(edges)
        related = [value for edge, value in cka.items() if edge in selected]
        controls = [value for edge, value in cka.items() if edge not in selected]
        return statistics.fmean(related) - statistics.fmean(controls)

    observed = statistic(relation_edges)
    rng = random.Random(seed)
    exceed = 0
    task_list = list(tasks)
    for _ in range(permutations_count):
        shuffled = task_list.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(tasks, shuffled))
        edges = [frozenset((mapping[relation.left], mapping[relation.right])) for relation in relations]
        if statistic(edges) >= observed:
            exceed += 1
    return {
        "statistic": "mean_direct_relation_cka_minus_mean_other_pair_cka",
        "observed": observed,
        "permutations": permutations_count,
        "one_sided_p": (exceed + 1) / (permutations_count + 1),
        "seed": seed,
    }


def _marker_and_checkpoint(run: GeometryRun) -> tuple[dict[str, Any], Path]:
    marker_path = Path(run.output_dir) / "completed.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(marker["checkpoint"]))
    if not checkpoint.is_file():
        checkpoint = marker_path.parent / checkpoint.name
    if _sha256(checkpoint) != marker["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for {run.run_id}")
    return marker, checkpoint


def _run_mapping(run: GeometryRun) -> dict[str, Any]:
    marker, checkpoint = _marker_and_checkpoint(run)
    return {
        "run_id": run.canonical_run_id,
        "architecture": run.architecture,
        "task_count": len(run.tasks),
        "seed": run.seed,
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "checkpoint_path": str(checkpoint),
    }


def _validate_checkpoint_signatures(
    runs: Sequence[GeometryRun], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    reference: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for run in runs:
        marker, checkpoint_path = _marker_and_checkpoint(run)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint_config = checkpoint.get("config")
        state = checkpoint.get("state")
        if not isinstance(checkpoint_config, Mapping) or not isinstance(state, Mapping):
            raise ValueError(f"checkpoint schema invalid for {run.run_id}")
        signature = {field: checkpoint_config.get(field) for field in SIGNATURE_FIELDS}
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"training signature differs for {run.run_id}")
        if (
            tuple(checkpoint_config.get("tasks", ())) != run.tasks
            or int(checkpoint_config.get("seed", -1)) != run.seed
            or int(state.get("global_step", -1)) != int(config["training"]["max_steps"])
        ):
            raise ValueError(f"task, seed, or step mismatch for {run.run_id}")
        validation = checkpoint.get("validation")
        if not isinstance(validation, Mapping) or not validation:
            raise ValueError(f"validation metrics missing for {run.run_id}")
        seen = [validation[task] for task in run.tasks]
        rows.append(
            {
                "logical_run_id": run.run_id,
                "canonical_run_id": run.canonical_run_id,
                "kind": run.kind,
                "split_id": run.split_id or "",
                "role": run.role,
                "related_pair_count": ""
                if run.related_pair_count is None
                else run.related_pair_count,
                "seed": run.seed,
                "tasks": ",".join(run.tasks),
                "reused": run.reused,
                "global_step": state["global_step"],
                "seen_task_token_accuracy_macro": statistics.fmean(
                    float(value["token_accuracy"]) for value in seen
                ),
                "seen_task_sequence_accuracy_macro": statistics.fmean(
                    float(value["sequence_accuracy"]) for value in seen
                ),
                "checkpoint_sha256": marker["checkpoint_sha256"],
            }
        )
        del checkpoint
    return rows


def _activation(
    run: GeometryRun,
    *,
    examples: Sequence[ProbeExample],
    probe_sha256: str,
    cache_dir: Path,
    repository: Path,
    device: torch.device,
    batch_size: int,
) -> ActivationSet:
    return _load_trained_activations(
        _run_mapping(run),
        repository=repository,
        examples=examples,
        probe_sha256=probe_sha256,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
    )


def _cka(a: ActivationSet, b: ActivationSet, layer: str, device: torch.device) -> float:
    value = float(linear_cka(a.layers[layer].to(device), b.layers[layer].to(device)).cpu())
    if not math.isfinite(value):
        raise ValueError("CKA is not finite")
    return value


def _specialist_rows(
    activations: Mapping[tuple[str, int], ActivationSet],
    relations: Sequence[TaskRelation],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    tasks = tuple(task for relation in relations for task in (relation.left, relation.right))
    related = {frozenset((relation.left, relation.right)) for relation in relations}
    layers = tuple(next(iter(activations.values())).layers)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed_a, seed_b in combinations(MODEL_SEEDS, 2):
            a = activations[task, seed_a]
            b = activations[task, seed_b]
            for layer in layers:
                rows.append(
                    {
                        "comparison": "same_task",
                        "task_a": task,
                        "task_b": task,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "same_seed": False,
                        "layer": layer,
                        "linear_cka": _cka(a, b, layer, device),
                    }
                )
    for task_a, task_b in combinations(tasks, 2):
        comparison = (
            "direct_relation"
            if frozenset((task_a, task_b)) in related
            else "no_direct_relation"
        )
        for seed_a, seed_b in product(MODEL_SEEDS, repeat=2):
            a = activations[task_a, seed_a]
            b = activations[task_b, seed_b]
            for layer in layers:
                rows.append(
                    {
                        "comparison": comparison,
                        "task_a": task_a,
                        "task_b": task_b,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "same_seed": seed_a == seed_b,
                        "layer": layer,
                        "linear_cka": _cka(a, b, layer, device),
                    }
                )
    return rows


def _bundle_rows(
    activations: Mapping[str, ActivationSet],
    runs: Sequence[GeometryRun],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bundle_runs = [run for run in runs if run.kind == "bundle"]
    for split_id in BUNDLE_SPLIT_IDS:
        for seed in MODEL_SEEDS:
            cell = [run for run in bundle_runs if run.split_id == split_id and run.seed == seed]
            anchor = next(run for run in cell if run.role == "anchor")
            a = activations[anchor.run_id]
            for count in RELATED_PAIR_COUNTS:
                other = next(run for run in cell if run.related_pair_count == count)
                b = activations[other.run_id]
                for layer in a.layers:
                    rows.append(
                        {
                            "split_id": split_id,
                            "model_seed": seed,
                            "related_pair_count": count,
                            "layer": layer,
                            "linear_cka": _cka(a, b, layer, device),
                            "anchor_run_id": anchor.run_id,
                            "other_run_id": other.run_id,
                        }
                    )
    return rows


def _symmetry_rows(
    identity: Mapping[tuple[str, int], ActivationSet],
    transformed: Mapping[tuple[str, int, str], ActivationSet],
    relations: Sequence[TaskRelation],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation in relations:
        wrong = "complement" if relation.input_transform == "inverse" else "inverse"
        for seed in MODEL_SEEDS:
            left = identity[relation.left, seed]
            variants = {
                "identity": identity[relation.right, seed],
                "correct": transformed[relation.right, seed, relation.input_transform],
                "wrong": transformed[relation.right, seed, wrong],
            }
            for condition, right in variants.items():
                for layer in left.layers:
                    rows.append(
                        {
                            "pair_id": relation.pair_id,
                            "task_left": relation.left,
                            "task_right": relation.right,
                            "model_seed": seed,
                            "condition": condition,
                            "applied_transform": "identity"
                            if condition == "identity"
                            else relation.input_transform
                            if condition == "correct"
                            else wrong,
                            "layer": layer,
                            "linear_cka": _cka(left, right, layer, device),
                        }
                    )
    return rows


def _symmetry_summary(
    rows: Sequence[Mapping[str, Any]], *, layer: str = "final_norm"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final = [row for row in rows if row["layer"] == layer]
    summaries: list[dict[str, Any]] = []
    for pair_id in sorted({str(row["pair_id"]) for row in final}):
        for condition in ("identity", "correct", "wrong"):
            values = [
                float(row["linear_cka"])
                for row in final
                if row["pair_id"] == pair_id and row["condition"] == condition
            ]
            summaries.append(
                {
                    "pair_id": pair_id,
                    "condition": condition,
                    "seed_count": len(values),
                    "final_layer_cka_mean": statistics.fmean(values),
                    "final_layer_cka_sample_sd": _sample_sd(values),
                }
            )
    by_unit = {
        (str(row["pair_id"]), int(row["model_seed"]), str(row["condition"])): float(
            row["linear_cka"]
        )
        for row in final
    }
    units = sorted({(pair, seed) for pair, seed, _ in by_unit})
    correct_identity = [
        by_unit[pair, seed, "correct"] - by_unit[pair, seed, "identity"]
        for pair, seed in units
    ]
    correct_wrong = [
        by_unit[pair, seed, "correct"] - by_unit[pair, seed, "wrong"]
        for pair, seed in units
    ]
    pairs = sorted({pair for pair, _ in units})
    relation_mean_deltas = {
        pair: {
            "correct_minus_identity": statistics.fmean(
                by_unit[pair, seed, "correct"] - by_unit[pair, seed, "identity"]
                for relation, seed in units
                if relation == pair
            ),
            "correct_minus_wrong": statistics.fmean(
                by_unit[pair, seed, "correct"] - by_unit[pair, seed, "wrong"]
                for relation, seed in units
                if relation == pair
            ),
        }
        for pair in pairs
    }
    relation_correct_identity = [
        relation_mean_deltas[pair]["correct_minus_identity"] for pair in pairs
    ]
    relation_correct_wrong = [
        relation_mean_deltas[pair]["correct_minus_wrong"] for pair in pairs
    ]

    def exact_sign_test(values: Sequence[float]) -> float:
        positives = sum(value > 0 for value in values)
        tail = sum(
            math.comb(len(values), index)
            for index in range(
                max(positives, len(values) - positives), len(values) + 1
            )
        )
        return min(1.0, 2 * tail / (2 ** len(values)))

    return summaries, {
        "pair_seed_units": len(units),
        "correct_minus_identity_mean": statistics.fmean(correct_identity),
        "correct_minus_identity_sample_sd": _sample_sd(correct_identity),
        "correct_minus_wrong_mean": statistics.fmean(correct_wrong),
        "correct_minus_wrong_sample_sd": _sample_sd(correct_wrong),
        "positive_correct_minus_identity_units": sum(value > 0 for value in correct_identity),
        "positive_correct_minus_wrong_units": sum(value > 0 for value in correct_wrong),
        "two_sided_exact_sign_test_p_correct_minus_identity": exact_sign_test(
            correct_identity
        ),
        "two_sided_exact_sign_test_p_correct_minus_wrong": exact_sign_test(
            correct_wrong
        ),
        "relation_units": len(pairs),
        "relation_mean_deltas": relation_mean_deltas,
        "positive_relation_mean_correct_minus_identity": sum(
            value > 0 for value in relation_correct_identity
        ),
        "positive_relation_mean_correct_minus_wrong": sum(
            value > 0 for value in relation_correct_wrong
        ),
        "two_sided_relation_level_sign_test_p_correct_minus_identity": exact_sign_test(
            relation_correct_identity
        ),
        "two_sided_relation_level_sign_test_p_correct_minus_wrong": exact_sign_test(
            relation_correct_wrong
        ),
    }


def _render_readme(
    specialist_groups: Sequence[Mapping[str, Any]],
    bundle_summary: Sequence[Mapping[str, Any]],
    bundle_trend: Mapping[str, Any],
    symmetry: Mapping[str, Any],
    permutation_test: Mapping[str, Any],
    random_rows: Sequence[Mapping[str, Any]],
) -> str:
    random_final = [
        float(row["linear_cka"])
        for row in random_rows
        if row["layer"] == "final_norm"
    ]
    lines = [
        "# Combinatorial task-geometry CKA results",
        "",
        "All values below use the same 4,096 task-free validation prefixes. ",
        "The test split was not read.",
        "",
        "## Single-task geometry",
        "",
        "| Comparison | Task-pair units | Final-layer CKA, mean +/- sample SD |",
        "|---|---:|---:|",
    ]
    for row in specialist_groups:
        lines.append(
            f"| {row['comparison']} | {row['task_pair_count']} | "
            f"{float(row['final_layer_cka_mean']):.6f} +/- "
            f"{float(row['final_layer_cka_sample_sd']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Task-label permutation contrast (direct minus other): "
            f"{float(permutation_test['observed']):+.6f}; one-sided "
            f"p={float(permutation_test['one_sided_p']):.6f}.",
            f"Random-initialization final-layer CKA across seeds: "
            f"{statistics.fmean(random_final):.6f} +/- "
            f"{_sample_sd(random_final):.6f}.",
            "Directly related tasks are more similar than the other cross-task "
            "pairs, but the absolute direct-relation CKA is far below the "
            "same-task value. The high random-initialization baseline reflects "
            "shared architecture and input geometry, not learned task structure.",
            "",
            "## Fixed-four-task composition",
            "",
            "| Direct correspondences r | Cells | Final-layer CKA, mean +/- sample SD |",
            "|---:|---:|---:|",
        ]
    )
    for row in bundle_summary:
        lines.append(
            f"| {row['related_pair_count']} | {row['cell_count']} | "
            f"{float(row['final_layer_cka_mean']):.6f} +/- "
            f"{float(row['final_layer_cka_sample_sd']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Mean paired r=4 minus r=0 delta: "
            f"{float(bundle_trend['delta_r4_minus_r0_mean']):+.6f} +/- "
            f"{float(bundle_trend['delta_r4_minus_r0_sample_sd']):.6f}.",
            f"Positive cells: {bundle_trend['positive_delta_cells']}/12; "
            f"monotonic cells: {bundle_trend['monotonic_cell_count']}/12.",
            f"Two-sided exact sign-test p for r=4 minus r=0: "
            f"{float(bundle_trend['two_sided_exact_sign_test_p']):.6f}.",
            "This controlled experiment does not support a monotonic CKA "
            "dose-response as direct correspondences increase: the effect is "
            "strongly heterogeneous across bundle layouts.",
            "",
            "## Symmetry-aligned mechanism",
            "",
            f"Correct minus identity CKA: "
            f"{float(symmetry['correct_minus_identity_mean']):+.6f} +/- "
            f"{float(symmetry['correct_minus_identity_sample_sd']):.6f}.",
            f"Correct minus wrong-transform CKA: "
            f"{float(symmetry['correct_minus_wrong_mean']):+.6f} +/- "
            f"{float(symmetry['correct_minus_wrong_sample_sd']):.6f}.",
            f"Descriptively, both contrasts are positive in "
            f"{symmetry['positive_correct_minus_identity_units']}/"
            f"{symmetry['pair_seed_units']} and "
            f"{symmetry['positive_correct_minus_wrong_units']}/"
            f"{symmetry['pair_seed_units']} pair-seed units. Treating those "
            "units as independent would give two-sided exact sign-test p-values of "
            f"{float(symmetry['two_sided_exact_sign_test_p_correct_minus_identity']):.3e} "
            f"and {float(symmetry['two_sided_exact_sign_test_p_correct_minus_wrong']):.3e}; "
            "these are descriptive because seeds are clustered within relations.",
            f"After averaging over seeds, both contrasts remain positive in "
            f"{symmetry['positive_relation_mean_correct_minus_identity']}/"
            f"{symmetry['relation_units']} and "
            f"{symmetry['positive_relation_mean_correct_minus_wrong']}/"
            f"{symmetry['relation_units']} mathematical relations. The primary "
            f"relation-level two-sided exact sign-test p-values are "
            f"{float(symmetry['two_sided_relation_level_sign_test_p_correct_minus_identity']):.6f} "
            f"and {float(symmetry['two_sided_relation_level_sign_test_p_correct_minus_wrong']):.6f}.",
            "The strongest result is therefore transformation-specific: models "
            "trained on known dual properties align when their inputs are related "
            "by the corresponding combinatorial symmetry.",
            "",
            "CKA is a representation diagnostic, not a behavioral accuracy metric or "
            "proof of a shared algorithm. See the raw CSV files for every model, "
            "task pair, seed, condition, and layer.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def run_task_geometry_cka(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    matrix = geometry_summary(config_path)
    if matrix["complete_count"] != matrix["run_count"]:
        raise ValueError("all 108 task-geometry models must complete before CKA")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    runs = build_geometry_matrix(config_path)
    relations = relation_definitions(config_path)
    provenance_rows = _validate_checkpoint_signatures(runs, config)
    repository = Path.cwd()
    validation_manifest = Path(config["validation_manifest"])
    probe_count = int(config["analysis"]["probe_examples"])
    probe_seed = int(config["analysis"]["probe_seed"])
    identity_examples = select_probe_examples(
        validation_manifest, count=probe_count, seed=probe_seed, shard_index=0
    )
    probes: dict[str, tuple[ProbeExample, ...]] = {
        "identity": identity_examples,
        "inverse": transform_probe_examples(identity_examples, "inverse"),
        "complement": transform_probe_examples(identity_examples, "complement"),
    }
    probe_manifests = {
        name: transformed_probe_identity(
            examples,
            transformation=name,
            dataset_manifest_sha256=_sha256(validation_manifest),
            shard_index=0,
            seed=probe_seed,
        )
        for name, examples in probes.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(probe_manifests, output_dir / "probe_manifests.json")

    identity_activations: dict[str, ActivationSet] = {}
    for index, run in enumerate(runs, start=1):
        print(f"extracting identity activation {index}/108: {run.run_id}", flush=True)
        identity_activations[run.run_id] = _activation(
            run,
            examples=identity_examples,
            probe_sha256=probe_manifests["identity"]["probe_sha256"],
            cache_dir=output_dir / "cache" / "identity",
            repository=repository,
            device=device,
            batch_size=batch_size,
        )
    specialists = [run for run in runs if run.kind == "specialist"]
    specialist_identity = {
        (run.tasks[0], run.seed): identity_activations[run.run_id] for run in specialists
    }
    transformed_activations: dict[tuple[str, int, str], ActivationSet] = {}
    for transformation in ("inverse", "complement"):
        for index, run in enumerate(specialists, start=1):
            print(
                f"extracting {transformation} specialist activation {index}/48: "
                f"{run.run_id}",
                flush=True,
            )
            transformed_activations[run.tasks[0], run.seed, transformation] = _activation(
                run,
                examples=probes[transformation],
                probe_sha256=probe_manifests[transformation]["probe_sha256"],
                cache_dir=output_dir / "cache" / transformation,
                repository=repository,
                device=device,
                batch_size=batch_size,
            )

    specialist_rows = _specialist_rows(specialist_identity, relations, device=device)
    specialist_pair_rows, specialist_group_rows = summarize_specialist_rows(specialist_rows)
    permutation_test = task_label_permutation_test(
        specialist_pair_rows,
        relations,
        permutations_count=int(config["analysis"]["task_pair_permutations"]),
        seed=probe_seed,
    )
    bundle_rows = _bundle_rows(identity_activations, runs, device=device)
    bundle_summary, bundle_trend = summarize_bundle_rows(bundle_rows)
    symmetry_rows = _symmetry_rows(
        specialist_identity, transformed_activations, relations, device=device
    )
    symmetry_summary, symmetry_trend = _symmetry_summary(symmetry_rows)

    reference = _run_mapping(runs[0])
    random_activations = {
        seed: _load_random_activations(
            reference,
            repository=repository,
            seed=seed,
            examples=identity_examples,
            probe_sha256=probe_manifests["identity"]["probe_sha256"],
            cache_dir=output_dir / "cache" / "random",
            device=device,
            batch_size=batch_size,
        )
        for seed in MODEL_SEEDS
    }
    random_rows = [
        {
            "seed_a": seed_a,
            "seed_b": seed_b,
            "layer": layer,
            "linear_cka": _cka(
                random_activations[seed_a], random_activations[seed_b], layer, device
            ),
        }
        for seed_a, seed_b in combinations(MODEL_SEEDS, 2)
        for layer in random_activations[seed_a].layers
    ]

    artifacts: dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[str]]] = {
        "run_provenance.csv": (provenance_rows, tuple(provenance_rows[0])),
        "specialist_pairwise_cka.csv": (specialist_rows, tuple(specialist_rows[0])),
        "specialist_pair_summary.csv": (
            specialist_pair_rows,
            tuple(specialist_pair_rows[0]),
        ),
        "specialist_group_summary.csv": (
            specialist_group_rows,
            tuple(specialist_group_rows[0]),
        ),
        "bundle_cell_cka.csv": (bundle_rows, tuple(bundle_rows[0])),
        "bundle_summary.csv": (bundle_summary, tuple(bundle_summary[0])),
        "symmetry_cka.csv": (symmetry_rows, tuple(symmetry_rows[0])),
        "symmetry_summary.csv": (symmetry_summary, tuple(symmetry_summary[0])),
        "random_init_cka.csv": (random_rows, tuple(random_rows[0])),
    }
    for name, (rows, fields) in artifacts.items():
        _atomic_csv(rows, fields, output_dir / name)
    readme = _render_readme(
        specialist_group_rows,
        bundle_summary,
        bundle_trend,
        symmetry_trend,
        permutation_test,
        random_rows,
    )
    _atomic_text(readme, output_dir / "README.md")
    manifest = {
        "status": "completed",
        "protocol_version": "property-task-geometry-cka/v1",
        "analysis_commit": _git_commit(repository),
        "config_path": str(config_path),
        "config_sha256": matrix["config_sha256"],
        "model_count": len(runs),
        "specialist_model_count": len(specialists),
        "bundle_model_count": len(runs) - len(specialists),
        "probe_manifests": probe_manifests,
        "specialist_task_label_permutation_test": permutation_test,
        "bundle_trend": bundle_trend,
        "symmetry_trend": symmetry_trend,
        "test_split_used": False,
        "artifacts": {
            **{name: _sha256(output_dir / name) for name in artifacts},
            "README.md": _sha256(output_dir / "README.md"),
            "probe_manifests.json": _sha256(output_dir / "probe_manifests.json"),
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    result = run_task_geometry_cka(
        args.config, args.output_dir, device=device, batch_size=args.batch_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "main",
    "run_task_geometry_cka",
    "summarize_bundle_rows",
    "summarize_specialist_rows",
    "task_label_permutation_test",
    "transform_probe_examples",
    "transformed_probe_identity",
]

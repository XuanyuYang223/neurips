from __future__ import annotations

from pathlib import Path

import pytest

from neurips_permutations.category_linear_probe import (
    ARCHITECTURE_LAYERS,
    CONDITIONS,
    FAMILY_NAMES,
    build_contrast_rows,
    build_macro_rows,
    property_family,
    summarize_macro_rows,
    validate_config,
)
from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES
from neurips_permutations.property_linear_probe import METRICS


REPOSITORY = Path(__file__).parents[1]


def _metric_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for architecture_index, architecture in enumerate(("transformer", "mlp")):
        for model_kind, conditions in (
            ("trained", CONDITIONS),
            ("random", ("random_init",)),
        ):
            for condition in conditions:
                for seed_index, seed in enumerate((17, 42, 314159)):
                    for layer_index, layer in enumerate(ARCHITECTURE_LAYERS[architecture]):
                        for task_index, task in enumerate(PROPERTY32_TASK_NAMES):
                            base = (
                                architecture_index
                                + seed_index
                                + layer_index
                                + task_index / 100
                            )
                            if model_kind == "trained":
                                base += 10
                            rows.append(
                                {
                                    "model_kind": model_kind,
                                    "run_id": f"{model_kind}-{architecture}-{condition}-{seed}",
                                    "architecture": architecture,
                                    "condition": condition,
                                    "model_seed": seed,
                                    "trained_tasks": "",
                                    "probe_task": task,
                                    "probe_task_family": property_family(task),
                                    "layer": layer,
                                    **{metric: base for metric in METRICS},
                                }
                            )
    return rows


def test_property_families_are_an_exact_partition() -> None:
    grouped = {name: [] for name in FAMILY_NAMES}
    for task in PROPERTY32_TASK_NAMES:
        grouped[property_family(task)].append(task)
    assert {name: len(tasks) for name, tasks in grouped.items()} == {
        "local": 8,
        "positional": 8,
        "cycle": 8,
        "global_run": 8,
    }
    with pytest.raises(ValueError, match="unknown scalar property"):
        property_family("not_a_property")


def test_macro_summary_and_paired_random_contrasts_are_complete() -> None:
    macros = build_macro_rows(_metric_rows())
    model_layer_count = 12 * len(ARCHITECTURE_LAYERS["transformer"]) + 12 * len(
        ARCHITECTURE_LAYERS["mlp"]
    )
    assert len(macros) == model_layer_count * 5
    assert {int(row["probe_task_count"]) for row in macros} == {8, 32}

    summary = summarize_macro_rows(macros)
    summary_layer_count = 4 * len(ARCHITECTURE_LAYERS["transformer"]) + 4 * len(
        ARCHITECTURE_LAYERS["mlp"]
    )
    assert len(summary) == summary_layer_count * 5
    contrasts = build_contrast_rows(macros)
    assert len(contrasts) == 3 * sum(map(len, ARCHITECTURE_LAYERS.values())) * 5
    assert all(int(row["seed_count"]) == 3 for row in contrasts)
    for row in contrasts:
        for metric in METRICS:
            assert float(row[f"{metric}_delta_mean"]) == pytest.approx(10.0)
            assert float(row[f"{metric}_delta_sample_sd"]) == pytest.approx(0.0)


def test_summary_rejects_a_missing_seed() -> None:
    macros = build_macro_rows(_metric_rows())
    damaged = [
        row
        for row in macros
        if not (
            row["model_kind"] == "trained"
            and row["architecture"] == "transformer"
            and row["condition"] == "encoding_e4"
            and int(row["model_seed"]) == 17
        )
    ]
    with pytest.raises(ValueError, match="lacks three seeds"):
        summarize_macro_rows(damaged)


def test_frozen_real_config_is_valid() -> None:
    value, digest = validate_config(REPOSITORY / "configs/v3_category_linear_probe.toml")
    assert value["probe"]["validation_examples"] == 8192
    assert value["probe"]["test_examples"] == 8192
    assert len(digest) == 64

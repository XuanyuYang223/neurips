from pathlib import Path

from neurips_permutations.passage import PERMUTATION20_VOCABULARY, TOKEN_TO_ID
from neurips_permutations.scaling_k16 import (
    ARCHITECTURES,
    CONDITION_ORDER,
    MODEL_SEEDS,
    _factorial_effect_rows,
    _paper_section,
    _scaling_figure_svg,
    plan,
)
from neurips_permutations.training import TrainConfig, _model_vocab_size


def test_k16_scaling_plan_is_frozen_and_balanced() -> None:
    value = plan(Path("configs/permutation_scaling_k16.toml"))
    assert value["run_count"] == 24
    assert value["new_run_count"] == 12
    rows = value["runs"]
    assert len({(row["condition"], row["architecture"], row["seed"]) for row in rows}) == 24
    assert {row["seed"] for row in rows} == {17, 42, 314159}
    assert {row["architecture"] for row in rows} == {"transformer", "mlp"}


def test_v3_manifest_recovers_original_vocab_after_property_extension() -> None:
    v3 = TrainConfig(
        output_dir="unused",
        manifest="data/permutation-10m-v3/manifest.json",
    )
    property32 = TrainConfig(
        output_dir="unused",
        manifest="data/permutation-properties-16m-v1/manifest.json",
    )
    assert len(PERMUTATION20_VOCABULARY) == 163
    assert len(TOKEN_TO_ID) == 188
    assert _model_vocab_size(v3) == 163
    assert _model_vocab_size(property32) == 188


def test_factorial_effects_are_seed_paired_for_loss_and_accuracy() -> None:
    rows = []
    offsets = {
        "baseline": 0.0,
        "data10x_model1x": 1.0,
        "data1x_model2x": 2.0,
        "data10x_model2x": 4.0,
    }
    for architecture in ARCHITECTURES:
        for condition in CONDITION_ORDER:
            for seed_index, seed in enumerate(MODEL_SEEDS):
                rows.append(
                    {
                        "condition": condition,
                        "architecture": architecture,
                        "seed": seed,
                        "loss": 10.0 + seed_index - offsets[condition],
                        "accuracy": seed_index + offsets[condition],
                    }
                )
    effects = _factorial_effect_rows(rows, ("loss", "accuracy"))
    assert len(effects) == 2 * 2 * 5
    lookup = {
        (row["architecture"], row["metric"], row["contrast"]): row
        for row in effects
    }
    assert lookup[("transformer", "loss", "data_effect_at_1x_model")]["mean"] == -1.0
    assert lookup[("transformer", "accuracy", "data_effect_at_1x_model")]["mean"] == 1.0
    assert lookup[("mlp", "accuracy", "data_by_model_interaction")]["mean"] == 1.0


def test_scaling_figure_and_paper_section_cover_full_grid() -> None:
    summary = []
    rows = []
    for condition_index, condition in enumerate(CONDITION_ORDER):
        for architecture in ARCHITECTURES:
            summary.append(
                {
                    "condition": condition,
                    "architecture": architecture,
                    "data_multiplier": 10 if "data10x" in condition else 1,
                    "model_multiplier": 2 if "model2x" in condition else 1,
                    "structured_holdout_loss_mean": 2.0 - 0.1 * condition_index,
                    "structured_holdout_loss_sample_sd": 0.1,
                    "structured_holdout_token_accuracy_mean": 0.2 + 0.01 * condition_index,
                    "structured_holdout_token_accuracy_sample_sd": 0.02,
                    "structured_holdout_sequence_accuracy_mean": 0.01 * condition_index,
                    "structured_holdout_sequence_accuracy_sample_sd": 0.005,
                    "parity_sequence_accuracy_mean": 0.1,
                    "parity_sequence_accuracy_sample_sd": 0.01,
                }
            )
            for seed in MODEL_SEEDS:
                rows.append(
                    {
                        "condition": condition,
                        "architecture": architecture,
                        "seed": seed,
                        "structured_holdout_loss": 2.0 - 0.1 * condition_index,
                        "structured_holdout_token_accuracy": 0.2 + 0.01 * condition_index,
                        "structured_holdout_sequence_accuracy": 0.01 * condition_index,
                    }
                )
    effects = _factorial_effect_rows(
        rows,
        (
            "structured_holdout_loss",
            "structured_holdout_token_accuracy",
            "structured_holdout_sequence_accuracy",
        ),
    )
    svg = _scaling_figure_svg(summary)
    paper = _paper_section(summary, effects)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert svg.count("<rect ") == 11
    assert "Transformer" in svg and "MLP" in svg
    assert "## Methods" in paper
    assert "## Results" in paper
    assert "## Limitations" in paper

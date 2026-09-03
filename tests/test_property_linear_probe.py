from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurips_permutations.cka import ActivationSet, ProbeExample
from neurips_permutations.math_ops import PROPERTY32_TASK_NAMES, PROPERTY_FUNCTIONS
from neurips_permutations.passage import TOKEN_TO_ID
from neurips_permutations.property_linear_probe import (
    LAYERS,
    METRICS,
    _probe_rows_for_activation_pair,
    _completed_result,
    _rank,
    _validate_config,
    build_primary_replicate_rows,
    build_run_macro_rows,
    decode_probe_permutation,
    fit_probe_layer,
    fit_target_normalizer,
    property_label_matrix,
    summarize_primary_rows,
    validation_train_tune_indices,
)


def _probe(record_id: int, permutation: tuple[int, ...]) -> ProbeExample:
    tokens = ["<BOS>", "<ONE_START>"]
    for index, value in enumerate(permutation):
        if index:
            tokens.append(",")
        tokens.append(f"{value:02d}")
    tokens.append("<ONE_END>")
    return ProbeExample(
        record_id=record_id,
        n=len(permutation),
        token_ids=tuple(TOKEN_TO_ID[token] for token in tokens),
    )


def test_frozen_config_and_property_labels_are_recomputed() -> None:
    config, digest = _validate_config(Path("configs/property32_linear_probe.toml"))
    assert config["probe"]["primary_task_status"] == "opposite_pool"
    assert len(digest) == 64

    examples = (_probe(10, (2, 1, 3)), _probe(11, (3, 1, 2)))
    assert decode_probe_permutation(examples[0]) == (2, 1, 3)
    labels = property_label_matrix(examples)
    assert labels.shape == (2, 32)
    for row, permutation in zip(labels, ((2, 1, 3), (3, 1, 2))):
        expected = [PROPERTY_FUNCTIONS[task](permutation) for task in PROPERTY32_TASK_NAMES]
        assert row.tolist() == expected


def test_completed_result_is_idempotent_and_authenticates_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "results.csv"
    artifact.write_text("header\n", encoding="utf-8")
    from neurips_permutations.cka import _sha256

    manifest = {
        "status": "completed",
        "protocol_version": "property32-linear-probe/v1",
        "config_sha256": "c" * 64,
        "test_split_used": True,
        "artifacts": {artifact.name: _sha256(artifact)},
    }
    (tmp_path / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )
    assert _completed_result(tmp_path, "c" * 64) == manifest
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        _completed_result(tmp_path, "c" * 64)


def test_validation_partition_is_deterministic_disjoint_and_length_stratified() -> None:
    examples = tuple(
        _probe(1_000 + index, tuple(range(1, n + 1)))
        for n in range(2, 8)
        for index in range(n * 10, n * 10 + 12)
    )
    first = validation_train_tune_indices(examples, tune_fraction=0.25, seed=17)
    second = validation_train_tune_indices(examples, tune_fraction=0.25, seed=17)
    changed = validation_train_tune_indices(examples, tune_fraction=0.25, seed=18)
    assert first == second
    assert first != changed
    train, tune = map(set, first)
    assert not train & tune
    assert train | tune == set(range(len(examples)))
    for n in range(2, 8):
        members = {index for index, example in enumerate(examples) if example.n == n}
        assert len(members & train) == 9
        assert len(members & tune) == 3


def _synthetic_probe_data(count: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.arange(offset, offset + count, dtype=torch.long)
    lengths = 2 + indices.remainder(3)
    first = indices.remainder(2).to(torch.float64)
    second = indices.div(2, rounding_mode="floor").remainder(2).to(torch.float64)
    labels = torch.stack((lengths.to(torch.float64) + first, 2 * lengths + second), dim=1)
    features = torch.stack(
        (
            first,
            second,
            lengths.to(torch.float64),
            (first - second) * 0.25,
        ),
        dim=1,
    )
    return features, labels, lengths


def test_ridge_probe_recovers_length_conditioned_integer_targets() -> None:
    validation_x, validation_y, validation_n = _synthetic_probe_data(240)
    test_x, test_y, test_n = _synthetic_probe_data(120, offset=1_000)
    results = fit_probe_layer(
        validation_x,
        test_x,
        validation_y,
        test_y,
        validation_n,
        test_n,
        tuple(range(180)),
        tuple(range(180, 240)),
        (1e-8, 1e-4, 1e-1),
    )
    assert len(results) == 2
    for result in results:
        assert result["length_conditioned_r2"] > 0.999
        assert result["pearson_r"] > 0.999
        assert result["exact_accuracy"] == 1.0
        assert result["exact_accuracy_minus_baseline"] > 0.45


def test_target_normalizer_uses_smallest_mode_for_ties() -> None:
    labels = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float64)
    lengths = torch.tensor([2, 2, 3, 3])
    normalizer = fit_target_normalizer(labels, lengths)
    assert normalizer.means[2, 0] == 0.5
    assert normalizer.means[3, 0] == 0.5
    assert normalizer.modes[2, 0] == 0.0
    assert normalizer.modes[3, 0] == 0.0


def _raw_row(
    *, replicate: str, pool: str, k: int, status: str, layer: str, value: float
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_kind": "trained",
        "replicate_id": replicate,
        "pool": pool,
        "run_id": f"{replicate}-{pool}-{k}",
        "model_seed": {"r0": 17, "r1": 42, "r2": 101}[replicate],
        "trained_task_count": k,
        "trained_tasks": "task",
        "probe_task": "descents",
        "probe_task_family": "local",
        "task_status": status,
        "layer": layer,
        "selected_ridge_alpha": 0.1,
        "tuning_length_conditioned_r2": value,
        "probe_train_examples": 3_072,
        "probe_tune_examples": 1_024,
        "probe_test_examples": 4_096,
    }
    row.update({metric: value for metric in METRICS})
    return row


def test_primary_summary_keeps_replicates_as_statistical_units() -> None:
    raw = []
    for replicate_index, replicate in enumerate(("r0", "r1", "r2")):
        for pool in ("a", "b"):
            for k in (1, 2, 4, 8, 16):
                for layer in LAYERS:
                    for task_index in range(16):
                        row = _raw_row(
                            replicate=replicate,
                            pool=pool,
                            k=k,
                            status="opposite_pool",
                            layer=layer,
                            value=k / 100 + replicate_index / 10,
                        )
                        row["probe_task"] = f"task-{task_index:02d}"
                        raw.append(row)
    macros = build_run_macro_rows(raw)
    assert len(macros) == 3 * 2 * 5 * 6
    assert {row["probe_task_count"] for row in macros} == {16}
    replicates = build_primary_replicate_rows(macros)
    assert len(replicates) == 3 * 5 * 6
    summary = summarize_primary_rows(replicates)
    assert len(summary) == 5 * 6
    final_k4 = next(
        row for row in summary if row["trained_task_count"] == 4 and row["layer"] == "final_norm"
    )
    assert final_k4["replicate_count"] == 3
    assert final_k4["length_conditioned_r2_mean"] == pytest.approx(0.14)
    assert final_k4["length_conditioned_r2_sample_sd"] == pytest.approx(0.1)


def test_probe_row_grid_labels_trained_same_pool_and_opposite_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = (_probe(1, (1, 2)), _probe(2, (2, 1)))
    layers = {name: torch.ones((2, 3)) for name in LAYERS}
    activation = ActivationSet(
        run_id="run",
        architecture="transformer",
        task_count=1,
        seed=17,
        checkpoint_sha256="a" * 64,
        probe_sha256="b" * 64,
        layers=layers,
    )

    monkeypatch.setattr(
        "neurips_permutations.property_linear_probe.fit_probe_layer",
        lambda *args, **kwargs: [
            {
                "selected_ridge_alpha": 0.1,
                "tuning_length_conditioned_r2": 0.0,
                **{metric: 0.0 for metric in METRICS},
            }
            for _ in PROPERTY32_TASK_NAMES
        ],
    )
    labels = property_label_matrix(examples)
    rows = _probe_rows_for_activation_pair(
        activation,
        activation,
        validation_labels=labels,
        test_labels=labels,
        validation_lengths=torch.tensor([2, 2]),
        test_lengths=torch.tensor([2, 2]),
        train_indices=(0,),
        tune_indices=(1,),
        alphas=(0.1,),
        metadata={
            "model_kind": "trained",
            "replicate_id": "r0",
            "pool": "a",
            "run_id": "run",
            "model_seed": 17,
            "trained_task_count": 1,
        },
        trained_tasks=(PROPERTY32_TASK_NAMES[0],),
        pool_tasks=PROPERTY32_TASK_NAMES[:16],
    )
    assert len(rows) == 6 * 32
    final = [row for row in rows if row["layer"] == "final_norm"]
    assert final[0]["task_status"] == "trained"
    assert final[1]["task_status"] == "same_pool_untrained"
    assert final[16]["task_status"] == "opposite_pool"


def test_average_rank_handles_ties() -> None:
    assert _rank((10.0, 20.0, 20.0, 30.0)) == [0.0, 1.5, 1.5, 3.0]

"""Numerical, extraction, sampling, and aggregation tests for CKA."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch

from neurips_permutations.cka import (
    ActivationSet,
    ProbeExample,
    build_pairwise_rows,
    build_category_pairwise_rows,
    extract_landmark_activations,
    linear_cka,
    probe_identity,
    select_probe_examples,
    summarize_pairwise_rows,
    summarize_category_rows,
    _spearman,
)
from neurips_permutations.models import ModelConfig, build_model
from neurips_permutations.passage import TOKEN_TO_ID


def _gram_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.double()
    y = y.double()
    gram_x = x @ x.T
    gram_y = y @ y.T
    n = x.shape[0]
    center = torch.eye(n, dtype=torch.float64) - torch.ones(
        n, n, dtype=torch.float64
    ) / n
    centered_x = center @ gram_x @ center
    centered_y = center @ gram_y @ center
    return (centered_x * centered_y).sum() / torch.sqrt(
        centered_x.square().sum() * centered_y.square().sum()
    )


def test_linear_cka_matches_centered_gram_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(31, 7, generator=generator)
    y = torch.randn(31, 11, generator=generator)
    torch.testing.assert_close(linear_cka(x, y), _gram_reference(x, y))


def test_linear_cka_self_orthogonal_scale_and_translation_invariance() -> None:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(64, 8, generator=generator)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=generator))
    assert abs(float(linear_cka(x, x)) - 1.0) < 1e-12
    assert abs(float(linear_cka(x, 3.5 * x @ q + 12.0)) - 1.0) < 1e-12


def test_linear_cka_rejects_invalid_or_constant_inputs() -> None:
    x = torch.ones(8, 3)
    for y in (torch.ones(8, 2), torch.ones(7, 3), torch.ones(8)):
        try:
            linear_cka(x, y)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid CKA input was accepted")


def test_spearman_reports_monotonic_and_reversed_trends() -> None:
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def _tiny_examples() -> tuple[ProbeExample, ...]:
    tokens = (
        TOKEN_TO_ID["<BOS>"],
        TOKEN_TO_ID["<SIZE>"],
        TOKEN_TO_ID["03"],
        TOKEN_TO_ID["<ONE_START>"],
        TOKEN_TO_ID["02"],
        TOKEN_TO_ID[","],
        TOKEN_TO_ID["01"],
        TOKEN_TO_ID[","],
        TOKEN_TO_ID["03"],
        TOKEN_TO_ID["<ONE_END>"],
    )
    return (
        ProbeExample(1, 3, tokens),
        ProbeExample(2, 3, tokens[:-4] + (TOKEN_TO_ID["03"],) + tokens[-3:]),
        ProbeExample(3, 3, tokens[:-2] + (TOKEN_TO_ID["02"], tokens[-1])),
        ProbeExample(4, 3, tokens),
    )


def test_hidden_extractor_returns_landmarks_without_changing_logits() -> None:
    examples = _tiny_examples()
    for model_type, layers in (("transformer", 2), ("mlp", 1)):
        config = ModelConfig(
            vocab_size=len(TOKEN_TO_ID),
            max_seq_len=32,
            d_model=16,
            layers=layers,
            model_type=model_type,
            n_heads=4 if model_type == "transformer" else None,
        )
        torch.manual_seed(5)
        model = build_model(config).eval()
        row = torch.tensor([examples[0].token_ids])
        with torch.no_grad():
            before = model(row)
        activations = extract_landmark_activations(
            model, examples, device=torch.device("cpu"), batch_size=3
        )
        with torch.no_grad():
            after = model(row)
        assert tuple(activations) == (
            "embedding",
            *(f"block_{index + 1:02d}" for index in range(layers)),
            "final_norm",
        )
        assert all(value.shape == (4, 16) for value in activations.values())
        torch.testing.assert_close(before, after)
        assert not any(module._forward_hooks for module in model.modules())


def test_probe_selection_is_deterministic_and_strips_task_answer(tmp_path: Path) -> None:
    shard = tmp_path / "part-00000.jsonl.gz"
    rows = []
    for record_id in range(20):
        tokens = [
            "<BOS>",
            "<SIZE>",
            "02",
            "<ONE_START>",
            "01",
            ",",
            "02",
            "<ONE_END>",
            "<LENGTH>",
            "=",
            "<ONE_START>",
            "02",
            ",",
            "01",
            "<ONE_END>",
            "<EOS>",
        ]
        rows.append({"id": record_id, "n": 2, "tokens": tokens})
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"shards": [{"filename": shard.name}]}))
    first = select_probe_examples(manifest, count=7, seed=9, shard_index=0)
    second = select_probe_examples(manifest, count=7, seed=9, shard_index=0)
    assert first == second
    assert all(example.token_ids[-1] == TOKEN_TO_ID["<ONE_END>"] for example in first)
    identity = probe_identity(
        first, dataset_manifest_sha256="a" * 64, shard_index=0, seed=9
    )
    assert identity["example_count"] == 7
    assert identity["n_histogram"] == {"2": 7}
    assert len(identity["probe_sha256"]) == 64


def _activation(
    architecture: str, task_count: int, seed: int, value: torch.Tensor
) -> ActivationSet:
    layers = {"embedding": value, "final_norm": value}
    return ActivationSet(
        run_id=f"{architecture}-{task_count}-{seed}",
        architecture=architecture,
        task_count=task_count,
        seed=seed,
        checkpoint_sha256="x",
        probe_sha256="p",
        layers=layers,
    )


def test_pair_matrix_and_summary_have_expected_primary_counts() -> None:
    generator = torch.Generator().manual_seed(12)
    trained = []
    random = []
    for architecture in ("transformer", "mlp"):
        for task_count in (1, 16):
            for seed in (17, 42, 314159):
                trained.append(
                    _activation(
                        architecture,
                        task_count,
                        seed,
                        torch.randn(12, 5, generator=generator),
                    )
                )
        for seed in (17, 42, 314159):
            random.append(
                _activation(
                    architecture,
                    0,
                    seed,
                    torch.randn(12, 5, generator=generator),
                )
            )
    rows = build_pairwise_rows(trained, random, compute_device=torch.device("cpu"))
    assert sum(row["comparison"] == "seed_stability" for row in rows) == 24
    assert sum(row["comparison"] == "k16_alignment" for row in rows) == 12
    assert sum(row["comparison"] == "cross_architecture" for row in rows) == 6
    assert sum(row["comparison"] == "random_baseline" for row in rows) == 12
    summaries = summarize_pairwise_rows(rows)
    primary = [row for row in summaries if row["comparison"] == "seed_stability"]
    assert len(primary) == 8
    assert all(row["pair_count"] == 3 for row in primary)


def test_category_matrix_separates_overlap_and_seed_relation() -> None:
    generator = torch.Generator().manual_seed(81)
    conditions = ("encoding_e4", "statistics_s4", "algebra_a4")
    seeds = (17, 42, 314159)
    trained = []
    condition_by_run = {}
    random = []
    for architecture in ("transformer", "mlp"):
        for condition in conditions:
            for seed in seeds:
                item = _activation(
                    architecture,
                    4,
                    seed,
                    torch.randn(12, 5, generator=generator),
                )
                item = ActivationSet(
                    run_id=f"{condition}-{item.run_id}",
                    architecture=item.architecture,
                    task_count=item.task_count,
                    seed=item.seed,
                    checkpoint_sha256=item.checkpoint_sha256,
                    probe_sha256=item.probe_sha256,
                    layers=item.layers,
                )
                trained.append(item)
                condition_by_run[item.run_id] = condition
        for seed in seeds:
            random.append(
                _activation(
                    architecture,
                    0,
                    seed,
                    torch.randn(12, 5, generator=generator),
                )
            )
    rows = build_category_pairwise_rows(
        trained,
        condition_by_run,
        random,
        compute_device=torch.device("cpu"),
    )
    counts = {
        comparison: sum(row["comparison"] == comparison for row in rows)
        for comparison in {row["comparison"] for row in rows}
    }
    assert counts == {
        "within_condition_cross_seed": 36,
        "disjoint_condition_same_seed": 36,
        "disjoint_condition_cross_seed": 72,
        "random_cross_seed": 12,
    }
    detailed, overall = summarize_category_rows(rows)
    assert len(detailed) == 40
    assert len(overall) == 16
    assert all(row["pair_count"] > 0 for row in detailed + overall)

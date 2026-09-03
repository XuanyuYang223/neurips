from __future__ import annotations

from collections import Counter

import pytest

from neurips_permutations.property_fewshot import (
    REPLICATE_IDS,
    _base_runs,
    _family,
    _support_specs,
    build_plan,
    load_spec,
)
from neurips_permutations.property_fewshot_results import (
    CONTRASTS,
    METRICS,
    build_family_summary,
    build_random_summary,
    build_replicate_rows,
    build_summary,
)


def test_frozen_property_fewshot_design_is_balanced_and_uses_shard199() -> None:
    spec = load_spec()
    assert spec.test_source_shard_index == 199
    assert spec.test_shard_position == 1
    assert spec.expected_test_examples == 2_500
    assert spec.expected_runs == 144
    assert len(_support_specs(spec)) == 24
    for targets in spec.target_sets.values():
        assert len(targets) == 4
        assert {_family(task) for task in targets} == {
            "local",
            "positional",
            "cycle",
            "global_run",
        }


def test_property_fewshot_plan_has_120_unseen_warm_starts_and_24_controls() -> None:
    spec = load_spec()
    base = _base_runs(spec, strict=False)
    plan = build_plan(spec, base)
    assert len(plan) == len({run["run_id"] for run in plan}) == 144
    counts = Counter(run["initialization"] for run in plan)
    assert counts == {"pretrained": 120, "random": 24}
    base_by_id = {run["run_id"]: run for run in base}
    for run in plan:
        assert run["architecture"] == "transformer"
        if run["initialization"] == "pretrained":
            assert run["task"] not in base_by_id[run["base_run_id"]]["tasks"]
    for replicate_id in REPLICATE_IDS:
        selected = [run for run in plan if run["replicate_id"] == replicate_id]
        assert len(selected) == 48
        assert Counter(run["initialization"] for run in selected) == {
            "pretrained": 40,
            "random": 8,
        }


def _result_row(
    *, replicate_id: str, pool: str, k: int, family: str, value: float, initialization: str
) -> dict[str, object]:
    seed = {"r0": 17, "r1": 42, "r2": 101}[replicate_id]
    row: dict[str, object] = {
        "run_id": f"{initialization}-{replicate_id}-{pool}-{k}-{family}",
        "initialization": initialization,
        "replicate_id": replicate_id,
        "model_pool": pool,
        "base_run_id": "base" if initialization == "pretrained" else "",
        "base_trained_task_count": k if initialization == "pretrained" else 0,
        "seed": seed,
        "task": f"{family}-{pool}",
        "target_family": family,
    }
    row.update({metric: value for metric in METRICS})
    row.update({metric: value for metric in CONTRASTS})
    return row


def test_property_fewshot_summaries_keep_three_replicates_as_units() -> None:
    families = ("local", "positional", "cycle", "global_run")
    rows = []
    for replicate_index, replicate_id in enumerate(REPLICATE_IDS):
        for pool in ("a", "b"):
            for k in (1, 2, 4, 8, 16):
                for family in families:
                    rows.append(
                        _result_row(
                            replicate_id=replicate_id,
                            pool=pool,
                            k=k,
                            family=family,
                            value=k / 100 + replicate_index / 10,
                            initialization="pretrained",
                        )
                    )
            for family in families:
                rows.append(
                    _result_row(
                        replicate_id=replicate_id,
                        pool=pool,
                        k=0,
                        family=family,
                        value=replicate_index / 10,
                        initialization="random",
                    )
                )
    replicates = build_replicate_rows(rows)
    assert len(replicates) == 15
    summary = build_summary(replicates)
    assert len(summary) == 5
    k4 = next(row for row in summary if row["base_trained_task_count"] == 4)
    assert k4["sequence_accuracy_mean"] == pytest.approx(0.14)
    assert k4["sequence_accuracy_sample_sd"] == pytest.approx(0.1)
    family = build_family_summary(rows)
    assert len(family) == 20
    random = build_random_summary(rows)
    assert len(random) == 1
    assert random[0]["sequence_accuracy_mean"] == pytest.approx(0.1)
    assert random[0]["sequence_accuracy_sample_sd"] == pytest.approx(0.1)

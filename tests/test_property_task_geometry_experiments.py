from collections import Counter
from pathlib import Path

import pytest

from neurips_permutations.property_task_geometry_experiments import (
    BUNDLE_SPLIT_IDS,
    MODEL_SEEDS,
    RELATED_PAIR_COUNTS,
    build_geometry_matrix,
    geometry_summary,
    relation_definitions,
    run_geometry_matrix,
    transform_permutation,
    validate_geometry_config,
    verify_relation_identities,
)


def test_geometry_config_freezes_balanced_two_layer_design() -> None:
    design = validate_geometry_config()
    assert design["relation_count"] == 8
    assert len(design["selected_tasks"]) == 16
    assert design["specialist_run_count"] == 48
    assert design["bundle_run_count"] == 60
    assert design["run_count"] == 108
    assert design["reuse_count"] == 8


def test_geometry_matrix_contains_only_unique_task_seed_designs() -> None:
    runs = build_geometry_matrix()
    assert len(runs) == len({run.run_id for run in runs}) == 108
    assert len({(run.architecture, run.seed, run.tasks) for run in runs}) == 108
    assert Counter(run.kind for run in runs) == {"bundle": 60, "specialist": 48}
    assert {run.seed for run in runs} == set(MODEL_SEEDS)
    reused = [run for run in runs if run.reused]
    assert len(reused) == 8
    assert all(run.kind == "specialist" for run in reused)


def test_every_bundle_cell_has_exact_preregistered_relation_count() -> None:
    relations = relation_definitions()
    pair = {
        task: relation.pair_id
        for relation in relations
        for task in (relation.left, relation.right)
    }
    runs = [run for run in build_geometry_matrix() if run.kind == "bundle"]
    for split_id in BUNDLE_SPLIT_IDS:
        for seed in MODEL_SEEDS:
            cell = [run for run in runs if run.split_id == split_id and run.seed == seed]
            assert len(cell) == 5
            anchor = next(run for run in cell if run.role == "anchor")
            for count in RELATED_PAIR_COUNTS:
                other = next(run for run in cell if run.related_pair_count == count)
                assert not (set(anchor.tasks) & set(other.tasks))
                observed = len(
                    {pair[task] for task in anchor.tasks}
                    & {pair[task] for task in other.tasks}
                )
                assert observed == count


def test_each_condition_and_anchor_use_every_task_once_across_splits() -> None:
    relations = relation_definitions()
    expected = Counter(
        task for relation in relations for task in (relation.left, relation.right)
    )
    runs = [
        run
        for run in build_geometry_matrix()
        if run.kind == "bundle" and run.seed == MODEL_SEEDS[0]
    ]
    anchors = Counter(task for run in runs if run.role == "anchor" for task in run.tasks)
    assert anchors == expected
    for count in RELATED_PAIR_COUNTS:
        observed = Counter(
            task for run in runs if run.related_pair_count == count for task in run.tasks
        )
        assert observed == expected


def test_preregistered_relations_hold_exhaustively_through_six() -> None:
    result = verify_relation_identities(max_n=6)
    assert result == {
        "max_n": 6,
        "permutations_checked": 872,
        "relations_checked": 8,
    }


def test_transform_permutation_rejects_unknown_transform() -> None:
    assert transform_permutation((2, 1, 3), "identity") == (2, 1, 3)
    assert transform_permutation((2, 1, 3), "inverse") == (2, 1, 3)
    assert transform_permutation((2, 1, 3), "complement") == (2, 3, 1)
    with pytest.raises(ValueError, match="unknown permutation transformation"):
        transform_permutation((2, 1, 3), "rotate")


def test_dry_run_partitions_all_models_without_overlap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "neurips_permutations.property_task_geometry_experiments._completed",
        lambda *_args, **_kwargs: False,
    )
    commands: list[str] = []
    for index in range(7):
        run_geometry_matrix(worker_index=index, worker_count=7, dry_run=True)
        commands.extend(capsys.readouterr().out.splitlines())
    assert len(commands) == len(set(commands)) == 108
    assert all("neurips_permutations.training" in command for command in commands)


def test_geometry_summary_accepts_completed_partial_or_unstarted_runs() -> None:
    summary = geometry_summary()
    assert summary["run_count"] == 108
    assert summary["complete_count"] + summary["incomplete_count"] == 108
    assert summary["by_kind"]["specialist"]["run_count"] == 48
    assert summary["by_kind"]["bundle"]["run_count"] == 60


def test_geometry_outputs_are_separate_from_superseded_matrix() -> None:
    fresh = [run for run in build_geometry_matrix() if not run.reused]
    assert all(Path(run.output_dir).parts[:2] == ("runs", "property-task-geometry") for run in fresh)

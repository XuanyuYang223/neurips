from pathlib import Path

import pytest

from neurips_permutations import property_relation_experiments as relation_module
from neurips_permutations.property_relation_experiments import (
    MODEL_SEEDS,
    SPLIT_IDS,
    TASK_COUNTS,
    build_relation_matrix,
    relation_summary,
    run_relation_matrix,
    validate_relation_config,
)
from neurips_permutations.property_replicates import NATURAL_DUAL_PAIRS


def test_relation_design_has_nine_cells_and_no_selected_natural_duals() -> None:
    design = validate_relation_config()
    assert design["cell_count"] == 9
    assert design["run_count"] == 72
    for split in design["splits"].values():
        selected = set(split["pool_a"]) | set(split["pool_b"])
        assert not (set(split["pool_a"]) & set(split["pool_b"]))
        assert len(selected) == 16
        assert all((left in selected) != (right in selected) for left, right in NATURAL_DUAL_PAIRS)
        assert split["co_selected_natural_duals"] == 0
        assert split["conditional_cross_correlations"]["max"] < 0.4


def test_relation_matrix_has_complete_unique_nested_grid() -> None:
    runs = build_relation_matrix()
    assert len(runs) == len({run.run_id for run in runs}) == 72
    physical = [run for run in runs if not run.is_alias]
    aliases = [run for run in runs if run.is_alias]
    assert len(physical) == 60
    assert len(aliases) == 12
    assert len({(run.architecture, run.seed, run.tasks) for run in physical}) == 60
    assert all(run.canonical_run_id != run.run_id for run in aliases)
    assert all(run.canonical_output_dir != run.output_dir for run in aliases)
    assert {run.split_id for run in runs} == set(SPLIT_IDS)
    assert {run.seed for run in runs} == set(MODEL_SEEDS)
    assert {run.task_count for run in runs} == set(TASK_COUNTS)
    for split_id in SPLIT_IDS:
        for seed in MODEL_SEEDS:
            cell = [run for run in runs if run.split_id == split_id and run.seed == seed]
            assert len(cell) == 8
            for task_count in TASK_COUNTS:
                matched = [run for run in cell if run.task_count == task_count]
                assert {run.pool for run in matched} == {"a", "b"}
                assert not (set(matched[0].tasks) & set(matched[1].tasks))
                assert all(len(run.tasks) == task_count for run in matched)


def test_relation_cell_dry_run_emits_only_unique_commands(capsys: pytest.CaptureFixture[str]) -> None:
    run_relation_matrix(cell="s1:42", dry_run=True)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 6
    assert all("s1-seed42" in line for line in lines)
    assert all("property32_relation_controlled.toml" in line for line in lines)


def test_relation_full_dry_run_emits_sixty_unique_models(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Dry-run command coverage must not depend on completed runs in the
    # developer's local ignored ``runs/`` tree.
    monkeypatch.setattr(relation_module, "_completed", lambda *args, **kwargs: False)
    run_relation_matrix(dry_run=True)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 60
    task_seed_designs = {
        (run.architecture, run.seed, run.tasks)
        for run in build_relation_matrix()
        if not run.is_alias
    }
    assert len(task_seed_designs) == 60


def test_relation_summary_accepts_completed_or_unstarted_matrix() -> None:
    summary = relation_summary()
    assert summary["run_count"] == 72
    assert summary["physical_run_count"] == 60
    assert summary["alias_count"] == 12
    assert summary["complete_count"] + summary["incomplete_count"] == 72
    assert (
        summary["complete_physical_count"] + summary["incomplete_physical_count"]
        == 60
    )
    assert set(summary["cells"]) == {
        f"{split_id}:{seed}" for split_id in SPLIT_IDS for seed in MODEL_SEEDS
    }


def test_relation_runner_rejects_unknown_cell() -> None:
    with pytest.raises(ValueError, match="unknown relation-controlled cell"):
        run_relation_matrix(cell="s9:999", dry_run=True)


def test_relation_outputs_are_isolated_from_previous_runs() -> None:
    outputs = {Path(run.output_dir).parts[:2] for run in build_relation_matrix()}
    assert outputs == {("runs", "property32-relation-controlled")}

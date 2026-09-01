"""Tests for the zero-overlap property pilot runner."""

from __future__ import annotations

from pathlib import Path

from neurips_permutations.property_experiments import (
    build_property_matrix,
    main,
    matrix_summary,
)


CONFIG = Path("configs/property32_zero_overlap_pilot.toml")


def test_frozen_matrix_has_ten_disjoint_transformer_runs() -> None:
    runs = build_property_matrix(CONFIG)
    assert len(runs) == 10
    assert {run.pool for run in runs} == {"a", "b"}
    assert {run.task_count for run in runs} == {1, 2, 4, 8, 16}
    assert {run.architecture for run in runs} == {"transformer"}
    assert {run.seed for run in runs} == {17}
    assert len({run.run_id for run in runs}) == 10
    for task_count in (1, 2, 4, 8, 16):
        selected = [run for run in runs if run.task_count == task_count]
        assert len(selected) == 2
        assert set(selected[0].tasks).isdisjoint(selected[1].tasks)


def test_initial_summary_and_dry_run_are_complete(tmp_path, capsys) -> None:
    isolated = tmp_path / "pilot.toml"
    isolated.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            'output_dir = "runs/property32-zero-overlap-pilot"',
            f'output_dir = "{tmp_path / "runs"}"',
        ),
        encoding="utf-8",
    )
    summary = matrix_summary(isolated)
    assert summary["run_count"] == 10
    assert summary["complete_count"] == 0
    assert summary["incomplete_count"] == 10

    assert main(["--config", str(isolated), "--run", "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 10
    assert all("neurips_permutations.training" in line for line in lines)
    assert all("--resume" in line for line in lines)

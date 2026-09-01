from neurips_permutations.property_replicates import (
    combined_status,
    validate_replicate_design,
)


def test_frozen_replicate_design_is_balanced_and_changes_membership() -> None:
    design = validate_replicate_design()
    assert design["run_count"] == 30
    assert {
        key: value["model_seed"] for key, value in design["replicates"].items()
    } == {"r0": 17, "r1": 42, "r2": 101}
    assert {
        key: value["dual_pairs_across_pools"]
        for key, value in design["replicates"].items()
    } == {"r0": 16, "r1": 6, "r2": 8}
    assert all(set(sides) == {"a", "b"} for sides in design["property_memberships"].values())


def test_combined_status_has_thirty_unique_runs() -> None:
    status = combined_status()
    assert status["run_count"] == 30
    run_ids = [
        run["run_id"]
        for summary in status["replicates"].values()
        for run in summary["runs"]
    ]
    assert len(run_ids) == len(set(run_ids)) == 30

from pathlib import Path

from neurips_permutations.fewshot import _run_identity
from neurips_permutations.property_fewshot import _base_runs, build_plan
from neurips_permutations.property_fewshot_lr_sensitivity import _load, plan


def test_lr_sensitivity_fills_exactly_two_missing_factorial_cells() -> None:
    value = plan(Path("configs/property32_fewshot_lr_sensitivity.toml"))
    assert value["run_count"] == 144
    assert value["pretrained_count"] == 120
    assert value["random_count"] == 24


def test_lr_sensitivity_identity_binds_protocol_and_matched_lr() -> None:
    _, _, protocol_sha256, base, sensitivity = _load(
        Path("configs/property32_fewshot_lr_sensitivity.toml")
    )
    assert sensitivity.config_sha256 == protocol_sha256
    assert sensitivity.config_sha256 != base.config_sha256
    runs = build_plan(base, _base_runs(base, strict=False))
    pretrained = next(run for run in runs if run["initialization"] == "pretrained")
    random = next(run for run in runs if run["initialization"] == "random")
    common = {
        "support_sha256": "a" * 64,
        "support_ids": list(range(20)),
        "implementation_commit": "b" * 40,
        "format_version": "property32-fewshot-lr-sensitivity/v1",
    }
    assert _run_identity(sensitivity, pretrained, **common)["learning_rate"] == 3e-4
    assert _run_identity(sensitivity, random, **common)["learning_rate"] == 1e-5

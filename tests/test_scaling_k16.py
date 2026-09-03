from pathlib import Path

from neurips_permutations.passage import PERMUTATION20_VOCABULARY, TOKEN_TO_ID
from neurips_permutations.scaling_k16 import plan
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

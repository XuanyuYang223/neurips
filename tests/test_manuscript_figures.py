from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from neurips_permutations.manuscript_figures import (
    MANUSCRIPT_FIGURE_FILES,
    SOURCE_FILES,
    generate,
)


REPOSITORY = Path(__file__).parents[1]


def _digests(path: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in MANUSCRIPT_FIGURE_FILES
    }


def test_l4_probe_snapshot_contains_frozen_manuscript_values() -> None:
    source = REPOSITORY / "paper/figure_data/property32_l4_retrained_probe_summary.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    indexed = {
        (row["scope"], int(row["trained_task_count"]), row["layer"]): row
        for row in rows
    }
    assert len(rows) == 17
    assert float(indexed[("opposite_pool", 1, "final_norm")]["length_conditioned_r2_mean"]) == pytest.approx(0.2071324091824939)
    assert float(indexed[("opposite_pool", 8, "final_norm")]["length_conditioned_r2_mean"]) == pytest.approx(0.3076475278665786)
    assert float(indexed[("opposite_pool", 16, "block_04")]["exact_accuracy_mean"]) == pytest.approx(0.4954986572265625)
    assert float(indexed[("trained", 16, "block_04")]["exact_accuracy_mean"]) == pytest.approx(0.5168024698893229)
    assert float(indexed[("random_init", 0, "final_norm")]["length_conditioned_r2_mean"]) == pytest.approx(0.2147668706479519)


def test_generate_manuscript_figures_is_complete_and_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    output = tmp_path / "figures"

    first = generate(REPOSITORY, output)
    first_digests = _digests(output)
    second = generate(REPOSITORY, output)

    assert first == second
    assert first_digests == _digests(output)
    assert first["status"] == "completed"
    assert set(first["inputs"]) == set(SOURCE_FILES)
    assert set(first["outputs"]) == set(MANUSCRIPT_FIGURE_FILES)
    assert first["l4_retraining_source"]["commit"] == "b6a9d6eac4ec94263b29edba4611d70277c2bfb0"

    for name in MANUSCRIPT_FIGURE_FILES:
        path = output / name
        assert path.stat().st_size == first["outputs"][name]["bytes"]
        assert first_digests[name] == first["outputs"][name]["sha256"]
        if name.endswith(".svg"):
            root = ET.parse(path).getroot()
            assert root.tag.endswith("svg")
        else:
            from PIL import Image

            with Image.open(path) as image:
                assert image.format == "PNG"
                if "fig4" in name:
                    assert image.size == (2_400, 960)
                else:
                    assert image.size == (1_520, 940)

    assert json.loads((output / "manifest.json").read_text()) == first

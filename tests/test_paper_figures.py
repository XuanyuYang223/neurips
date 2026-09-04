from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from neurips_permutations.paper_figures import FIGURE_FILES, generate


REPOSITORY = Path(__file__).parents[1]


def _digests(path: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in FIGURE_FILES
    }


def test_generate_paper_figures_is_complete_and_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    output = tmp_path / "figures"

    first = generate(REPOSITORY, output)
    first_digests = _digests(output)
    second = generate(REPOSITORY, output)

    assert first == second
    assert first_digests == _digests(output)
    assert first["status"] == "completed"
    assert len(first["inputs"]) == 9
    assert set(first["outputs"]) == set(FIGURE_FILES)

    for name in FIGURE_FILES:
        path = output / name
        assert path.stat().st_size == first["outputs"][name]["bytes"]
        assert _digests(output)[name] == first["outputs"][name]["sha256"]
        if name.endswith(".svg"):
            root = ET.parse(path).getroot()
            assert root.tag.endswith("svg")
        else:
            from PIL import Image

            with Image.open(path) as image:
                assert image.format == "PNG"
                assert image.width >= 1_800
                expected_height = 900 if name.startswith("figure2_") else 840
                assert image.height == expected_height

    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted == first

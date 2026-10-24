"""with_head exports must not emit any asset name twice (#13500)."""

from collections import Counter
from pathlib import Path
import re

import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

pytestmark = pytest.mark.unit
BODIES = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPECS = ["full_body_spec_v1.json", "full_body_spec_anthro_driver.json"]


@pytest.mark.parametrize("name", SPECS)
def test_with_head_assets_unique_and_compile(name: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml, _ = exporter.export_full_body_mjcf(
        (BODIES / name).read_bytes(), with_head=True
    )
    for kind in ("texture", "material", "mesh"):
        names = re.findall(rf'<{kind}\b[^>]*?\bname="([^"]+)"', xml)
        dupes = [n for n, c in Counter(names).items() if c > 1]
        assert not dupes, f"duplicate {kind} names: {dupes}"
    mujoco.MjModel.from_xml_string(xml)

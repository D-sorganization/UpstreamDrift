"""Native OpenSim runtime tests for generated full-body anthropometric models (MS-40 #10339).

Verifies that the generated .osim models instantiate correctly inside the OpenSim SDK
runtime, validating 44 coordinates, BodySet, MarkerSet, ContactGeometrySet, and ConstraintSet.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest
from tests.helpers.opensim_support import require_opensim

HAS_OPENSIM = importlib.util.find_spec("opensim") is not None

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not HAS_OPENSIM, reason="OpenSim python bindings not installed"),
]

ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = (
    ROOT / "src" / "engines" / "physics_engines" / "opensim" / "models" / "generated"
)
DRIVER_OSIM = GENERATED_DIR / "full_body_anthro_driver.osim"
IRON_OSIM = GENERATED_DIR / "full_body_anthro_iron7.osim"


def test_full_body_anthro_driver_opensim_instantiation() -> None:
    """Validate full-body driver model loading and topology inside OpenSim SDK."""
    if not DRIVER_OSIM.exists():
        pytest.skip("Generated driver osim not present")
    opensim = require_opensim()

    model = opensim.Model(str(DRIVER_OSIM))
    model.initSystem()

    # 44 coordinates
    assert model.getNumCoordinates() == 44
    coords = model.getCoordinateSet()
    assert coords.getSize() == 44

    # 24 bodies (plus Ground)
    assert model.getNumBodies() == 24

    # 34 tour markers
    markers = model.getMarkerSet()
    assert markers.getSize() == 34

    # Contact geometries (ground plane + 6 foot spheres)
    contact_geoms = model.getContactGeometrySet()
    assert contact_geoms.getSize() >= 7

    # Constraints (dual-grip closure)
    constraints = model.getConstraintSet()
    assert constraints.getSize() >= 1


def test_full_body_anthro_iron7_opensim_instantiation() -> None:
    """Validate full-body iron7 model loading and topology inside OpenSim SDK."""
    if not IRON_OSIM.exists():
        pytest.skip("Generated iron7 osim not present")
    opensim = require_opensim()

    model = opensim.Model(str(IRON_OSIM))
    model.initSystem()

    assert model.getNumCoordinates() == 44
    assert model.getNumBodies() == 24
    assert model.getMarkerSet().getSize() == 34


@pytest.mark.parametrize("osim_path", [DRIVER_OSIM, IRON_OSIM], ids=["driver", "iron7"])
def test_committed_model_loads_without_missing_geometry(osim_path: Path) -> None:
    """Loading a committed model finds its club meshes (#13016)."""
    opensim = require_opensim()
    sink = opensim.StringLogSink()
    opensim.Logger.addSink(sink)
    try:
        opensim.Model(str(osim_path)).initSystem()
    finally:
        opensim.Logger.removeSink(sink)
    log = sink.getString()
    assert "Couldn't find file" not in log, log


def test_parity_adapter_builds_without_missing_geometry() -> None:
    """The physics-only parity adapter exports no club meshes to look up (#13016)."""
    opensim = require_opensim()
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        build_parity_adapter,
    )

    spec = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    sink = opensim.StringLogSink()
    opensim.Logger.addSink(sink)
    try:
        build_parity_adapter(spec.read_bytes())
    finally:
        opensim.Logger.removeSink(sink)
    log = sink.getString()
    assert "Couldn't find file" not in log, log

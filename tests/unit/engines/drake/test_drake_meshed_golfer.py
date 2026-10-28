"""Drake shows the v2 meshed golfer, not the primitive default (MeshCat fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydrake.multibody.plant")
pytest.importorskip("mujoco")

from src.engines.physics_engines.drake.python.src.drake_meshed_golfer import (  # noqa: E402
    build_meshed_golfer,
    drake_mujoco_parity,
)
from src.engines.physics_engines.drake.python.src.drake_ui_mixin import (  # noqa: E402
    DrakeUIMixin,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def test_gui_mixin_restores_manipulability_checkbox_population() -> None:
    """The GUI crashed at start-up (after MeshCat opened) without this method."""
    assert callable(getattr(DrakeUIMixin, "_populate_manip_checkboxes", None))


def test_meshed_golfer_registers_body_meshes_on_the_spec_bodies() -> None:
    golfer = build_meshed_golfer(SPEC.read_bytes(), meshcat=None)
    inspector = golfer.scene_graph.model_inspector()
    meshes = [
        gid
        for gid in inspector.GetAllGeometryIds()
        if inspector.GetIllustrationProperties(gid) is not None
        and type(inspector.GetShape(gid)).__name__ == "Mesh"
    ]
    assert len(meshes) >= 10
    assert golfer.plant.num_positions() >= len(golfer.coordinate_order)


def test_drake_matches_mujoco_for_bodies_joints_and_masses() -> None:
    report = drake_mujoco_parity(SPEC.read_bytes())
    assert report.missing_in_drake == () and report.missing_in_mujoco == ()
    assert report.max_body_mass_error_kg < 1e-9
    assert report.total_mass_error_kg < 1e-9


def test_parity_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        drake_mujoco_parity("not bytes")  # type: ignore[arg-type]

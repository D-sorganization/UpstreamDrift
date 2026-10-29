"""Renderer tests for capacity wireframe glyphs (#14045)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_overlay.capacity_glyphs import (
    CapacityWireGlyph,
    capacity_wire_segments,
)
from src.shared.python.force_overlay.renderers.capacity_wire_meshcat import (
    draw_capacity_wire_segments,
)

pytestmark = pytest.mark.unit


class _RecordingSink:
    def __init__(self) -> None:
        self.cylinders: list[tuple[str, float]] = []
        self.transforms: list[str] = []

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self.cylinders.append((path, length_m))

    def set_transform(self, path: str, matrix4x4: np.ndarray) -> None:
        self.transforms.append(path)

    def delete(self, path: str) -> None:
        return None


def test_meshcat_sink_receives_one_cylinder_per_segment():
    glyph = CapacityWireGlyph(
        centre=(0.0, 0.0, 0.0),
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        radii=(0.05, 0.03, 0.02),
        rgba=(0.8, 0.3, 0.1, 1.0),
    )
    segments = capacity_wire_segments(glyph, rim_samples=12)
    sink = _RecordingSink()
    paths = draw_capacity_wire_segments(sink, "overlay", segments)
    assert len(paths) == len(segments)
    assert len(sink.cylinders) == len(segments)


@pytest.mark.requires_mujoco
def test_mujoco_scene_accepts_capacity_wire():
    import mujoco

    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyphs import (
        add_capacity_wire_to_scene,
    )

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='1 1 0.1'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=512)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    glyph = CapacityWireGlyph(
        centre=(0.0, 0.0, 0.2),
        axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        radii=(0.04, 0.03, 0.02),
        rgba=(0.0, 0.45, 0.7, 0.9),
    )
    before = scene.ngeom
    receipt = add_capacity_wire_to_scene(scene, [glyph], rim_samples=10)
    assert receipt.added > 0
    assert scene.ngeom == before + receipt.added

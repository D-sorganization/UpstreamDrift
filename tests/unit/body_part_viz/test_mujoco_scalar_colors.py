"""MuJoCo scalar scene colouring smoke tests (#15396)."""

from __future__ import annotations

import pytest

from src.shared.python.body_part_viz.mujoco_scalar_colors import (
    apply_mujoco_scalar_scene_colors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

XML = """<mujoco><worldbody><body name="rod">
<geom type="capsule" size="0.05 0.2" rgba="1 1 0 1"/>
</body></worldbody></mujoco>"""


def _scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return model, scene


def test_scalar_shading_changes_scene_rgb():
    model, scene = _scene()
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    base = [float(v) for v in scene.geoms[0].rgba]
    apply_mujoco_scalar_scene_colors(model, scene, {"rod": 75.0}, scale)
    after = [float(v) for v in scene.geoms[0].rgba]
    assert after != pytest.approx(base)


def test_scalar_shading_marks_saturated_segments() -> None:
    model, scene = _scene()
    scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=10.0,
    )
    clipped: set[str] = set()
    apply_mujoco_scalar_scene_colors(
        model, scene, {"rod": 100.0}, scale, clipped_segments=clipped
    )
    assert "rod" in clipped

    clipped.clear()
    apply_mujoco_scalar_scene_colors(
        model, scene, {"rod": 5.0}, scale, clipped_segments=clipped
    )
    assert "rod" not in clipped

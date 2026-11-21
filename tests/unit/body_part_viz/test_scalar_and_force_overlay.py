"""Force and stored-KE overlays can be active together (#15396)."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from src.shared.python.body_part_viz import AxialLoadFrame, ForceColorScale  # noqa: E402
from src.shared.python.body_part_viz.mujoco_force_colors import (  # noqa: E402
    apply_mujoco_scene_colors,
)
from src.shared.python.body_part_viz.mujoco_scalar_colors import (  # noqa: E402
    apply_mujoco_scalar_scene_colors,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale  # noqa: E402

pytestmark = pytest.mark.unit

XML = """<mujoco><worldbody><body name="rod">
<geom type="capsule" size="0.05 0.2" rgba="0.8 0.2 0.2 1"/>
</body></worldbody></mujoco>"""


def _scene():
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


def test_force_and_scalar_overlays_both_mutate_scene_rgb() -> None:
    model, scene = _scene()
    base = [float(v) for v in scene.geoms[0].rgba[:3]]
    force_scale = ForceColorScale(
        enabled=True, tension_limit_n=1, compression_limit_n=1
    )
    apply_mujoco_scene_colors(
        model, scene, AxialLoadFrame(0.0, {"rod": 10.0}, "t"), force_scale
    )
    after_force = [float(v) for v in scene.geoms[0].rgba[:3]]
    assert after_force != pytest.approx(base)
    ke_scale = ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        colormap_id="cividis",
        low_limit=0.0,
        high_limit=100.0,
    )
    apply_mujoco_scalar_scene_colors(model, scene, {"rod": 50.0}, ke_scale)
    after_ke = [float(v) for v in scene.geoms[0].rgba[:3]]
    assert after_ke != pytest.approx(after_force)
